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

Outputs
  results/e3/agentdojo_gt_outputs.csv   per tool output: suite, user/injection task, tool, injected, score, p
  results/e3/agentdojo_items.csv        per (suite, llm, defense, seed, attack, user_task, injection_task): utility, security
  results/e3/agentdojo_guard_log/*.csv  guard decisions (output and drift) for tasks run in this process
"""
from __future__ import annotations

import argparse
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


def count_logged(logdir: Path, pipeline_name: str, suite: str, attack: str, prefix: str = "user_task") -> int:
    """Finished runs already in AgentDojo's logs for this pipeline / suite / attack ('none' = benign).
    prefix 'user_task' counts user-task runs; 'injection_task' counts the benign runs of injection tasks that
    AgentDojo does before an attack (to check that each injection goal is achievable)."""
    base = Path(logdir) / pipeline_name / suite
    return sum(1 for f in base.glob(f"{prefix}*/{attack}/*.json") if _finished(f)) if base.exists() else 0


def _finished(path: Path) -> bool:
    """AgentDojo also writes a log when a task is interrupted (API error, Ctrl+C): that log has no utility/security,
    AgentDojo re-runs the task on the next start, and it must not be counted as done."""
    import json
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(d.get("utility"), bool) and isinstance(d.get("security"), bool)


def reset_for_attack(pe, label, logdir, pipeline_name, suite, attack, aname, n_user):
    """Progress for benchmark_suite_with_injections: (unless DoS) every injection task once without injections,
    then every (user task, injection task) pair under the attack."""
    n_inj = len(suite.injection_tasks)
    pre = 0 if attack.is_dos_attack else n_inj
    done = (0 if attack.is_dos_attack else
            count_logged(logdir, pipeline_name, suite.name, "none", "injection_task")) + \
        count_logged(logdir, pipeline_name, suite.name, aname, "user_task")
    pe.reset(label, pre + n_user * n_inj, done)


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
    for lc in ac["llms"]:
        for seed in cfg["seeds"]:
            llm = make_llm(lc)
            mshort = lc["model"].split("/")[-1]
            # 1) calibration: benign runs on every suite's calibration tasks (resumable through AgentDojo's logs),
            #    then read back from the logs and pooled into one guard
            pe = ProgressElement()
            p_cal = build_pipeline(llm, f"{llm.name}-calib-seed{seed}", first=[pe])
            splits, outputs, pairs = {}, defaultdict(list), []
            for sname in ac["suites"]:
                suite = get_suite(bv, sname)
                cal_ids, test_ids = split_tasks(suite, ac["calibration_task_frac"], seed)
                splits[sname] = (suite, test_ids)
                pe.reset(f"E3 {mshort} seed {seed} calibration {sname}", len(cal_ids),
                         count_logged(LOGDIR, p_cal.name, sname, "none"))
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
                    for dfn in ac["defenses"]:
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
                        pipe.name = f"{llm.name}-{dfn}-seed{seed}"    # unique log folder per configuration
                        for aname in ["none", *ac["attacks"]]:
                            key = dict(llm=lc["model"], seed=seed, suite=sname, defense=dfn, attack=aname)
                            marker = out / "done" / f"agents_{mshort}_seed{seed}_{sname}_{dfn}_{aname}.ok"
                            if marker.exists():
                                print(f"[E3 {mshort} seed {seed} {sname} {dfn} {aname}] already done, skipping", flush=True)
                                continue
                            purge_rows(out / "agentdojo_items.csv", key)
                            label = f"E3 {mshort} seed {seed} {sname} {dfn} {aname}"
                            if aname == "none":
                                pe.reset(label, len(test_ids), count_logged(LOGDIR, pipe.name, sname, "none"))
                                res = benchmark_suite_without_injections(pipe, suite, LOGDIR, False, user_tasks=test_ids,
                                                                         benchmark_version=bv)
                                rows = [dict(**key, user_task=k[0], injection_task="", utility=u, security=None)
                                        for k, u in res["utility_results"].items()]
                            else:
                                attack = load_attack(aname, suite, pipe)
                                reset_for_attack(pe, label, LOGDIR, pipe.name, suite, attack, aname, len(test_ids))
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
                               pooled_can_flag=guard.pooled_can_flag)


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
