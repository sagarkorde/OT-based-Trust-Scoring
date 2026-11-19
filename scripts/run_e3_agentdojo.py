"""E3 (AgentDojo): tool-output scoring + intent drift inside AgentDojo's own benchmark loop (RQ3).

Modes
  --gt-detection   LLM-free: tool outputs from AgentDojo's ground-truth trajectories, with and without the
                   attack's injections. Gives tool-output detection AUROC/TPR/FPR on real AgentDojo data.
  (default)        Full agent runs with an LLM served behind an OpenAI-compatible API (vLLM / Ollama). Calibration
                   on benign runs of calibration user tasks, evaluation on the disjoint test user tasks.
  --smoke          --gt-detection with the test-only hashing encoder, written under results/_smoke/.

Resumable
  Agent runs: every task's full trajectory is saved by AgentDojo under results/agentdojo_logs/<pipeline>/...,
  and a restarted run reloads finished tasks instead of re-running them (logdir + force_rerun=False). Pipeline
  names carry model, defense and seed, so no two configurations share logs. Calibration is read back from those
  logs, so a resumed run calibrates on exactly the same benign trajectories. A finished (model, seed, suite,
  defense, attack) block is marked in results/e3/done/ and skipped outright.
  Ground-truth detection: one marker per seed.

Replay (config agentdojo.replay: true; authors' decision 2026-11-19)
  At temperature 0 a run depends only on what the model is shown. (1) The no-defense, spotlighting and PI-detector
  pipelines do not use the seed (the seed only picks the calibration/test split and calibrates the WTS guard), so
  their logs are shared by all seeds: each task runs once, whichever seed needs it first. Calibration runs use the
  no-defense pipeline (identical structure). (2) A WTS-defended run is identical to the no-defense run until a
  guard element acts. For each case the guard elements' own code is run on every prefix of the saved no-defense
  trajectory at the points where the live pipeline calls them; if no element would act, the no-defense result is
  copied (replayed=True); otherwise the case is run for real. (3) A random sample of copied cases is rerun for real
  with the WTS pipeline (results/e3/agentdojo_replay_check.csv).

Outputs
  results/e3/agentdojo_gt_outputs.csv   per tool output: suite, user/injection task, tool, injected, score, p
  results/e3/agentdojo_items.csv        per (suite, llm, defense, seed, attack, user_task, injection_task): utility, security
  results/e3/agentdojo_guard_log/*.csv  guard decisions (output and drift) for tasks run in this process
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agentdojo.agent_pipeline import GroundTruthPipeline  # noqa: E402
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

from wts.encoders import HashingEncoder, build_encoder  # noqa: E402
from wts.guards import AgentGuard  # noqa: E402
from wts.resume import Progress, mark_done, purge_rows  # noqa: E402
from wts.runlog import append_rows, refuse_test_only, run_record  # noqa: E402
from wts.stats import detection_metrics  # noqa: E402

LOGDIR = REPO / "results/agentdojo_logs"
DETERMINISTIC = ("none", "spotlighting_with_delimiting", "transformers_pi_detector")   # no seed-dependent part


def split_tasks(suite, frac, seed):
    ids = sorted(suite.user_tasks)
    rng = np.random.default_rng(seed)
    perm = [ids[i] for i in rng.permutation(len(ids))]
    k = int(round(frac * len(ids)))
    return perm[:k], perm[k:]


def _norm(s: str) -> str:
    return " ".join(s.replace("\\n", " ").split())


def contains_injection(text: str, injections: dict) -> bool:
    """AgentDojo's tool-result formatting collapses blank lines and may escape characters, so an exact
    substring test misses injected outputs. An output counts as injected if any substantive line
    (> 20 characters) of an injection appears in it after whitespace normalisation."""
    t = _norm(text)
    for v in injections.values():
        lines = [_norm(l) for l in (v or "").splitlines() if len(l.strip()) > 20]
        if any(l in t for l in lines):
            return True
    return False


def gt_tool_outputs(suite, user_task, injections):
    """Tool outputs of the user task's ground-truth trajectory in an environment carrying `injections`."""
    env = suite.load_and_inject_default_environment(injections)
    env = user_task.init_environment(env)
    gt = GroundTruthPipeline(user_task)
    _, _, _, msgs, _ = gt.query(user_task.PROMPT, FunctionsRuntime(suite.tools), env)
    return [(m["tool_call"].function, get_text_content_as_str(m["content"]) or "") for m in msgs if m["role"] == "tool"]


# ---------------------------------------------------------------------------------------- progress + logs
class ProgressElement(BasePipelineElement):
    """First element of a pipeline: AgentDojo calls the pipeline once per task it actually runs (finished tasks
    reloaded from the logs never reach it), so counting calls gives live progress in the console."""

    def __init__(self):
        self.p = None
        self.name = None

    def reset(self, label: str, total: int | None, resumed: int = 0):
        self.p = Progress(label, total or 0, resumed=resumed, every_s=60)

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if self.p is not None:
            self.p.step()
        return query, runtime, env, messages, extra_args


def count_logged(logdir: Path, pipeline_name: str, suite: str, attack: str, prefix: str = "user_task",
                 user_tasks=None) -> int:
    """Finished runs already in AgentDojo's logs for this pipeline / suite / attack ('none' = benign).
    prefix 'user_task' counts user-task runs; 'injection_task' counts the benign runs of injection tasks that
    AgentDojo does before an attack (to check that each injection goal is achievable)."""
    base = Path(logdir) / pipeline_name / suite
    if not base.exists():
        return 0
    keep = None if user_tasks is None else set(user_tasks)
    return sum(1 for f in base.glob(f"{prefix}*/{attack}/*.json")
               if (keep is None or f.parent.parent.name in keep) and _finished(f))


def _finished(path: Path) -> bool:
    """AgentDojo also writes a log when a task is interrupted (API error, Ctrl+C): that log has no utility/security,
    AgentDojo re-runs the task on the next start, and it must not be counted as done."""
    import json
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(d.get("utility"), bool) and isinstance(d.get("security"), bool)


def reset_for_attack(pe, label, logdir, pipeline_name, suite, attack, aname, n_user, user_tasks=None):
    """Progress for benchmark_suite_with_injections: (unless DoS) every injection task once without injections,
    then every (user task, injection task) pair under the attack."""
    n_inj = len(suite.injection_tasks)
    pre = 0 if attack.is_dos_attack else n_inj
    done = (0 if attack.is_dos_attack else
            count_logged(logdir, pipeline_name, suite.name, "none", "injection_task")) + \
        count_logged(logdir, pipeline_name, suite.name, aname, "user_task", user_tasks)
    pe.reset(label, pre + n_user * n_inj, min(done, pre + n_user * n_inj))


def harvest_calibration(logdir: Path, pipeline_name: str, suite_name: str, user_ids: list[str]):
    """Benign tool outputs (by tool) and consecutive plan-step pairs, read from the saved trajectories."""
    from agentdojo.benchmark import load_task_results

    from wts.agentdojo_hook import plan_text
    outputs, pairs = defaultdict(list), []
    for uid in user_ids:
        try:
            res = load_task_results(pipeline_name, suite_name, uid, "none", "none", logdir)
        except (FileNotFoundError, ValueError):   # ValueError covers pydantic's ValidationError: interrupted task
            continue
        steps = []
        for m in res.messages:
            if m["role"] == "tool" and m.get("tool_call") is not None:
                outputs[m["tool_call"].function].append(get_text_content_as_str(m["content"]) or "")
            elif m["role"] == "assistant":
                steps.append(plan_text(m))
        pairs += [(steps[i - 1], steps[i]) for i in range(1, len(steps))]
    return outputs, pairs


# ---------------------------------------------------------------------------------------- replay
def offline_guard_check(guard, enc, dfn: str, messages):
    """Would the WTS elements of `dfn` act anywhere on this no-defense trajectory? Runs the elements' own code on
    each prefix of the trajectory at the point where the live pipeline calls them: the drift element after every
    model step, the tool-output element after every block of tool results. Returns (acts, output_log, drift_log);
    the logs stop at the first action, after which the defended run would diverge."""
    from agentdojo.functions_runtime import EmptyEnv

    from wts.agentdojo_hook import WTSIntentDriftGuard, WTSToolOutputGuard
    n = 2 if dfn == "wts_full" else 1
    olog, dlog = [], []
    og = WTSToolOutputGuard(guard, enc, "block", n_tests=n, log=olog) \
        if dfn in ("wts_output_block", "wts_output_quarantine", "wts_full") else None
    dg = WTSIntentDriftGuard(guard, enc, n_tests=n, log=dlog) if dfn in ("wts_drift_confirm", "wts_full") else None
    msgs = copy.deepcopy(list(messages))
    for j, m in enumerate(msgs):
        if dg is not None and m["role"] == "assistant":
            dg.query("", None, EmptyEnv(), msgs[: j + 1], {})
            if any(e["flag"] for e in dlog):
                return True, olog, dlog
        if og is not None and m["role"] == "tool" and (j + 1 == len(msgs) or msgs[j + 1]["role"] != "tool"):
            og.query("", None, EmptyEnv(), msgs[: j + 1], {})
            if any(e["flag"] for e in olog):
                return True, olog, dlog
    return False, olog, dlog


def replay_block(pipe, suite, test_ids, aname, attack, guard, enc, dfn, none_name, key, bv, label, olog, dlog):
    """Rows for one (seed, suite, WTS defense, attack) block: copied from the no-defense run where no guard element
    would act, run for real otherwise (AgentDojo's own single-task runners, logged and resumable)."""
    from agentdojo.benchmark import load_task_results, run_task_with_injection_tasks, run_task_without_injection_tasks
    inj = ["none"] if aname == "none" else sorted(suite.injection_tasks)
    P = Progress(label, len(test_ids) * len(inj), every_s=60)
    rows, n_copy = [], 0
    for uid in test_ids:
        to_run = []
        for iid in inj:
            ref = load_task_results(none_name, suite.name, uid, aname, iid, LOGDIR)
            acts, ol, dl = offline_guard_check(guard, enc, dfn, ref.messages)
            tag = dict(user_task=uid, injection_task="" if aname == "none" else iid, replayed=not acts)
            olog += [dict(e, text=str(e.get("text", ""))[:300], **tag) for e in ol]
            dlog += [dict(e, plan=str(e.get("plan", ""))[:300], **tag) for e in dl]
            if acts:
                to_run.append(iid)
                continue
            rows.append(dict(**key, user_task=uid, injection_task=tag["injection_task"], utility=ref.utility,
                             security=None if aname == "none" else ref.security, replayed=True))
            n_copy += 1
            P.step(note="replayed")
        if not to_run:
            continue
        if aname == "none":
            u, _ = run_task_without_injection_tasks(suite, pipe, suite.user_tasks[uid], LOGDIR, False, bv)
            rows.append(dict(**key, user_task=uid, injection_task="", utility=u, security=None, replayed=False))
            P.step(note="run")
        else:
            ur, sr = run_task_with_injection_tasks(suite, pipe, suite.user_tasks[uid], attack, LOGDIR, False,
                                                   injection_tasks=to_run, benchmark_version=bv)
            for k, u in ur.items():
                rows.append(dict(**key, user_task=k[0], injection_task=k[1], utility=u, security=sr[k], replayed=False))
                P.step(note="run")
    print(f"[{label}] {n_copy} of {len(rows)} cases replayed from the no-defense run, {len(rows) - n_copy} run",
          flush=True)
    return rows


def replay_check(llm, lc, guard, enc, splits, bv, n, out, build_pipeline, WTSIntentDriftGuard, WTSToolOutputGuard):
    """Determinism check: rerun a random sample of replayed wts_full attack cases for real with the WTS pipeline
    and compare result and trajectory with the copied no-defense run."""
    import pandas as pd
    from agentdojo.benchmark import load_task_results, run_task_with_injection_tasks
    mshort = lc["model"].split("/")[-1]
    marker = out / "done" / f"agentdojo_replay_check_{mshort}.ok"
    if marker.exists():
        return
    items = pd.read_csv(out / "agentdojo_items.csv", dtype=str, keep_default_na=False)
    seed0 = str(min(int(x) for x in items.seed.unique()))
    cand = items[(items.llm == lc["model"]) & (items.seed == seed0) & (items.defense == "wts_full")
                 & (items.attack != "none") & (items.replayed.str.lower() == "true")]
    pick = cand.sample(min(n, len(cand)), random_state=0) if len(cand) else cand
    pipe = build_pipeline(llm, "", [WTSToolOutputGuard(guard, enc, "block", n_tests=2)],
                          [WTSIntentDriftGuard(guard, enc, n_tests=2)])
    pipe.name = f"{llm.name}-wts_full-seed{seed0}-check"
    none_name = f"{llm.name}-none"
    P = Progress(f"E3 {mshort} determinism check", len(pick), every_s=60)
    rows = []

    def traj(msgs):
        return [(m["role"], str(m.get("content")), str(m.get("tool_calls"))) for m in msgs]
    for r in pick.to_dict("records"):
        suite = splits[r["suite"]][0]
        attack = load_attack(r["attack"], suite, pipe)
        ur, sr = run_task_with_injection_tasks(suite, pipe, suite.user_tasks[r["user_task"]], attack, LOGDIR, False,
                                               injection_tasks=[r["injection_task"]], benchmark_version=bv)
        k = (r["user_task"], r["injection_task"])
        a = load_task_results(pipe.name, r["suite"], k[0], r["attack"], k[1], LOGDIR)
        b = load_task_results(none_name, r["suite"], k[0], r["attack"], k[1], LOGDIR)
        rows.append(dict(llm=lc["model"], suite=r["suite"], user_task=k[0], injection_task=k[1],
                         recorded_utility=r["utility"], rerun_utility=ur[k], recorded_security=r["security"],
                         rerun_security=sr[k], same_result=(str(ur[k]) == r["utility"] and str(sr[k]) == r["security"]),
                         same_trajectory=traj(a.messages) == traj(b.messages)))
        P.step()
    append_rows(out / "agentdojo_replay_check.csv", rows)
    mark_done(marker, dict(n=len(rows), same_result=sum(x["same_result"] for x in rows),
                           same_trajectory=sum(x["same_trajectory"] for x in rows)))
    print(f"[E3 {mshort}] determinism check: {sum(x['same_result'] for x in rows)}/{len(rows)} same result, "
          f"{sum(x['same_trajectory'] for x in rows)}/{len(rows)} identical trajectory", flush=True)


# ---------------------------------------------------------------------------------------- GT detection
def run_gt_detection(cfg, smoke=False):
    out = REPO / ("results/_smoke/e3" if smoke else "results/e3")
    ac = cfg["agentdojo"]
    enc = HashingEncoder(dim=64) if smoke else build_encoder(cfg["encoder"])
    if not smoke:
        refuse_test_only(enc)
    seeds = cfg["seeds"][:1] if smoke else cfg["seeds"]
    suites = ac["suites"]
    attacks = ac["attacks"][:1] if smoke else ac["attacks"]
    for seed in seeds:
        marker = out / "done" / f"gt_{enc.name}_seed{seed}.ok"
        if marker.exists() and not smoke:
            print(f"[E3 ground truth] seed {seed}: already done, skipping", flush=True)
            continue
        for f in ("agentdojo_gt_outputs.csv", "agentdojo_gt_detection.csv"):   # drop a partly written seed
            purge_rows(out / f, {"seed": seed})
        with run_record("_smoke/e3" if smoke else "e3", dict(cfg, mode="gt_detection", encoder=enc.name), seed,
                        REPO / "results") as rec:
            rows, det = [], []
            # one guard calibrated on benign outputs pooled over all suites' calibration tasks: a single suite
            # yields too few outputs to certify alpha/2 (conformal needs n_cal >= 39)
            benign, splits = defaultdict(list), {}
            for sname in suites:
                suite = get_suite(ac["benchmark_version"], sname)
                cal_ids, test_ids = split_tasks(suite, ac["calibration_task_frac"], seed)
                splits[sname] = (suite, test_ids)
                for uid in cal_ids:
                    for tool, text in gt_tool_outputs(suite, suite.user_tasks[uid], {}):
                        benign[tool].append(enc.encode(text))
            guard = AgentGuard(alpha=cfg["alpha"], min_tool_ref_units=5, ref_kwargs=dict(seed=seed))
            guard.fit_outputs(dict(benign), seed=seed)
            rec.update(n_benign_outputs=sum(len(v) for v in benign.values()), pooled_can_flag=guard.pooled_can_flag,
                       per_tool_models=sorted(k for k in guard.tool_refs if k != "__all__"))

            class _Named:  # attacks read the target model name from pipeline.name
                name = "local"
            P = Progress(f"E3 ground truth seed {seed}",
                         sum(len(t) for _, t in splits.values()) * len(attacks), every_s=30)
            for sname, (suite, test_ids) in splits.items():
                for aname in attacks:
                    attack = load_attack(aname, suite, _Named())
                    for uid in test_ids:
                        ut = suite.user_tasks[uid]
                        for tool, text in gt_tool_outputs(suite, ut, {}):
                            s, p = guard.output_pvalue(tool, enc.encode(text))
                            rows.append(dict(seed=seed, suite=sname, attack=aname, user_task=uid, injection_task="",
                                             tool=tool, injected=False, score=s, p=p))
                        for iid, it in suite.injection_tasks.items():
                            inj = attack.attack(ut, it)
                            for tool, text in gt_tool_outputs(suite, ut, inj):
                                injected = contains_injection(text, inj)
                                s, p = guard.output_pvalue(tool, enc.encode(text))
                                rows.append(dict(seed=seed, suite=sname, attack=aname, user_task=uid,
                                                 injection_task=iid, tool=tool, injected=injected, score=s, p=p))
                        P.step(note=sname)
            df = pd.DataFrame(rows)
            for (sname, aname), g in df.groupby(["suite", "attack"]):
                g = g[(g.injection_task == "") | g.injected]  # outputs that carry the injection vs clean outputs
                if g.injected.nunique() == 2:
                    m = detection_metrics(g.injected.astype(int).values, g.score.values)
                    flag = g.p <= cfg["alpha"] / 2
                    det.append(dict(seed=seed, suite=sname, attack=aname, encoder=enc.name, **m,
                                    tpr_at_alpha=float(flag[g.injected].mean()), fpr_at_alpha=float(flag[~g.injected].mean())))
            append_rows(out / "agentdojo_gt_outputs.csv", rows)
            append_rows(out / "agentdojo_gt_detection.csv", det)
            rec.update(n_outputs=len(rows), detection=det)
            for d in det:
                print(f"[E3 ground truth] seed={seed} {d['suite']}/{d['attack']}: AUROC={d['auroc']:.3f} "
                      f"n_pos={d['n_pos']} n_neg={d['n_neg']} TPR@a={d['tpr_at_alpha']:.2f} FPR@a={d['fpr_at_alpha']:.2f}",
                      flush=True)
        if not smoke:
            mark_done(marker, dict(n_outputs=len(rows)))


# ---------------------------------------------------------------------------------------- full agent runs
def make_llm(lc):
    import openai
    from agentdojo.agent_pipeline import LocalLLM, OpenAILLM
    if lc.get("ollama_base"):
        from wts.hw import ensure_ollama_tag
        ensure_ollama_tag(lc["model"], lc["ollama_base"], lc.get("num_ctx", 16384),
                          lc["base_url"].replace("/v1", ""), template_from=lc.get("template_from"))
    if lc.get("provider") == "openai":
        llm = OpenAILLM(openai.OpenAI(), lc["model"])
        llm.name = lc["model"]
    else:
        from wts.agentdojo_hook import text_tool_results_local_llm
        # AgentDojo's LocalLLM turns any request error into an empty reply, so a timeout silently fails the task:
        # allow 30 min per request (the OpenAI client default is 10 min; long agent contexts on a laptop GPU exceed it)
        llm = text_tool_results_local_llm(openai.OpenAI(base_url=lc["base_url"], api_key="EMPTY", timeout=1800),
                                          lc["model"])
        llm.name = f"local-{lc['model'].split('/')[-1]}"
    return llm


def run_agents(cfg):
    from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections

    from wts.agentdojo_hook import QuietLogger, WTSIntentDriftGuard, WTSToolOutputGuard, build_pipeline

    with QuietLogger(LOGDIR):          # AgentDojo saves each task's trajectory only inside a logger context
        _run_agents(cfg, AgentPipeline, PipelineConfig, benchmark_suite_with_injections,
                    benchmark_suite_without_injections, WTSIntentDriftGuard, WTSToolOutputGuard, build_pipeline)


def _run_agents(cfg, AgentPipeline, PipelineConfig, benchmark_suite_with_injections, benchmark_suite_without_injections,
                WTSIntentDriftGuard, WTSToolOutputGuard, build_pipeline):
    out = REPO / "results/e3"
    (out / "agentdojo_guard_log").mkdir(parents=True, exist_ok=True)
    ac = cfg["agentdojo"]
    enc = build_encoder(cfg["encoder"])
    refuse_test_only(enc)
    bv = ac["benchmark_version"]
    replay = bool(ac.get("replay", False))
    defenses = sorted(ac["defenses"], key=lambda d: (d != "none", d not in DETERMINISTIC))   # no-defense first
    if replay and "none" not in defenses:
        raise ValueError("agentdojo.replay needs the 'none' defense (it is the replay source)")
    for lc in ac["llms"]:
        for seed in cfg["seeds"]:
            llm = make_llm(lc)
            mshort = lc["model"].split("/")[-1]
            none_name = f"{llm.name}-none"
            # 1) calibration: benign runs on every suite's calibration tasks (resumable through AgentDojo's logs),
            #    then read back from the logs and pooled into one guard
            pe = ProgressElement()
            if replay:     # the no-defense pipeline, whose logs all seeds share
                p_cal = AgentPipeline.from_config(PipelineConfig(llm=llm, model_id=None, defense=None,
                                                                 system_message_name=None, system_message=None))
                p_cal.elements = [pe, *p_cal.elements]
                p_cal.name = none_name
            else:
                p_cal = build_pipeline(llm, f"{llm.name}-calib-seed{seed}", first=[pe])
            splits, outputs, pairs = {}, defaultdict(list), []
            for sname in ac["suites"]:
                suite = get_suite(bv, sname)
                cal_ids, test_ids = split_tasks(suite, ac["calibration_task_frac"], seed)
                splits[sname] = (suite, test_ids)
                pe.reset(f"E3 {mshort} seed {seed} calibration {sname}", len(cal_ids),
                         count_logged(LOGDIR, p_cal.name, sname, "none", user_tasks=cal_ids))
                benchmark_suite_without_injections(p_cal, suite, LOGDIR, False, user_tasks=cal_ids, benchmark_version=bv)
                o, pr = harvest_calibration(LOGDIR, p_cal.name, sname, cal_ids)
                for t, v in o.items():
                    outputs[t] += v
                pairs += pr
            guard = AgentGuard(alpha=cfg["alpha"], min_tool_ref_units=10, ref_kwargs=dict(seed=seed))
            guard.fit_outputs({t: [enc.encode(x) for x in v] for t, v in outputs.items()}, seed=seed)
            guard.fit_drift([(enc.encode(a), enc.encode(b)) for a, b in pairs], seed=seed)
            print(f"[E3 {mshort} seed {seed}] guard calibrated on {sum(len(v) for v in outputs.values())} benign "
                  f"tool outputs and {len(pairs)} plan-step pairs", flush=True)

            for sname, (suite, test_ids) in splits.items():
                with run_record("e3", dict(cfg, mode="agents", llm=lc["model"], suite=sname), seed) as rec:
                    for dfn in defenses:
                        pe = ProgressElement()
                        olog, dlog = [], []
                        if dfn in ("none", "spotlighting_with_delimiting", "transformers_pi_detector"):
                            pc = PipelineConfig(llm=llm, model_id=None, defense=None if dfn == "none" else dfn,
                                                system_message_name=None, system_message=None)
                            pipe = AgentPipeline.from_config(pc)
                            pipe.elements = [pe, *pipe.elements]
                        else:
                            n = 2 if dfn == "wts_full" else 1
                            after_tools, after_llm = [], []
                            if dfn in ("wts_output_block", "wts_output_quarantine", "wts_full"):
                                act = "quarantine" if dfn == "wts_output_quarantine" else "block"
                                after_tools.append(WTSToolOutputGuard(guard, enc, act, n_tests=n, log=olog))
                            if dfn in ("wts_drift_confirm", "wts_full"):
                                after_llm.append(WTSIntentDriftGuard(guard, enc, n_tests=n, log=dlog))
                            pipe = build_pipeline(llm, "", after_tools, after_llm, first=[pe])
                        # unique log folder per configuration; with replay the seed-free pipelines are shared by seeds
                        pipe.name = f"{llm.name}-{dfn}" if replay and dfn in DETERMINISTIC else f"{llm.name}-{dfn}-seed{seed}"
                        for aname in ["none", *ac["attacks"]]:
                            key = dict(llm=lc["model"], seed=seed, suite=sname, defense=dfn, attack=aname)
                            marker = out / "done" / f"agents_{mshort}_seed{seed}_{sname}_{dfn}_{aname}.ok"
                            if marker.exists():
                                print(f"[E3 {mshort} seed {seed} {sname} {dfn} {aname}] already done, skipping", flush=True)
                                continue
                            purge_rows(out / "agentdojo_items.csv", key)
                            label = f"E3 {mshort} seed {seed} {sname} {dfn} {aname}"
                            if replay and dfn not in DETERMINISTIC:
                                attack = None if aname == "none" else load_attack(aname, suite, pipe)
                                rows = replay_block(pipe, suite, test_ids, aname, attack, guard, enc, dfn, none_name,
                                                    key, bv, label, olog, dlog)
                                append_rows(out / "agentdojo_items.csv", rows)
                                mark_done(marker, dict(n_rows=len(rows)))
                                continue
                            if aname == "none":
                                pe.reset(label, len(test_ids), count_logged(LOGDIR, pipe.name, sname, "none",
                                                                            user_tasks=test_ids))
                                res = benchmark_suite_without_injections(pipe, suite, LOGDIR, False, user_tasks=test_ids,
                                                                         benchmark_version=bv)
                                rows = [dict(**key, user_task=k[0], injection_task="", utility=u, security=None)
                                        for k, u in res["utility_results"].items()]
                            else:
                                attack = load_attack(aname, suite, pipe)
                                reset_for_attack(pe, label, LOGDIR, pipe.name, suite, attack, aname, len(test_ids),
                                                 user_tasks=test_ids)
                                res = benchmark_suite_with_injections(pipe, suite, attack, LOGDIR, False,
                                                                      user_tasks=test_ids, benchmark_version=bv,
                                                                      verbose=False)
                                rows = [dict(**key, user_task=k[0], injection_task=k[1], utility=u,
                                             security=res["security_results"][k])
                                        for k, u in res["utility_results"].items()]
                            append_rows(out / "agentdojo_items.csv", rows)
                            mark_done(marker, dict(n_rows=len(rows)))
                        tag = f"{mshort}_{sname}_{dfn}_seed{seed}"
                        append_rows(out / "agentdojo_guard_log" / f"out_{tag}.csv", olog)
                        append_rows(out / "agentdojo_guard_log" / f"drift_{tag}.csv", dlog)
                    rec.update(n_cal_outputs=sum(len(v) for v in outputs.values()), n_plan_pairs=len(pairs),
                               pooled_can_flag=guard.pooled_can_flag, replay=replay)
            if replay and seed == cfg["seeds"][0] and "wts_full" in defenses:
                replay_check(llm, lc, guard, enc, splits, bv, int(ac.get("replay_check_n", 30)), out,
                             build_pipeline, WTSIntentDriftGuard, WTSToolOutputGuard)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/e3.yaml"))
    ap.add_argument("--gt-detection", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    if a.smoke or a.gt_detection:
        run_gt_detection(cfg, smoke=a.smoke)
    else:
        run_agents(cfg)
