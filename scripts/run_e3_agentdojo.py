"""E3 (AgentDojo): tool-output scoring + intent drift inside AgentDojo's own benchmark loop (RQ3).

Modes
  --gt-detection   LLM-free: tool outputs from AgentDojo's ground-truth trajectories, with and without the
                   attack's injections. Gives tool-output detection AUROC/TPR/FPR on real AgentDojo data.
  (default)        Full agent runs with an LLM served behind an OpenAI-compatible API (vLLM). Calibration on
                   benign runs of calibration user tasks, evaluation on the disjoint test user tasks.
  --smoke          --gt-detection with the test-only hashing encoder, written under results/_smoke/.

Outputs
  results/e3/agentdojo_gt_outputs.csv   per tool output: suite, user/injection task, tool, injected, score, p
  results/e3/agentdojo_items.csv        per (suite, llm, defense, seed, user_task, injection_task): utility, security
  results/e3/agentdojo_guard_log/*.csv.gz  every guard decision (output and drift)
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
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

from wts.encoders import HashingEncoder, build_encoder  # noqa: E402
from wts.guards import AgentGuard  # noqa: E402
from wts.runlog import append_rows, refuse_test_only, run_record  # noqa: E402
from wts.stats import detection_metrics  # noqa: E402


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
                print(f"[e3-gt] seed={seed} {d['suite']}/{d['attack']}: AUROC={d['auroc']:.3f} "
                      f"n_pos={d['n_pos']} n_neg={d['n_neg']} TPR@a={d['tpr_at_alpha']:.2f} FPR@a={d['fpr_at_alpha']:.2f}",
                      flush=True)


# ---------------------------------------------------------------------------------------- full agent runs
def make_llm(lc):
    import openai
    from agentdojo.agent_pipeline import LocalLLM, OpenAILLM
    if lc.get("ollama_base"):
        from wts.hw import ensure_ollama_tag
        ensure_ollama_tag(lc["model"], lc["ollama_base"], lc.get("num_ctx", 16384),
                          lc["base_url"].replace("/v1", ""))
    if lc.get("provider") == "openai":
        llm = OpenAILLM(openai.OpenAI(), lc["model"])
        llm.name = lc["model"]
    else:
        llm = LocalLLM(openai.OpenAI(base_url=lc["base_url"], api_key="EMPTY"), lc["model"])
        llm.name = f"local-{lc['model'].split('/')[-1]}"
    return llm


def run_agents(cfg):
    from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
    from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections

    from wts.agentdojo_hook import Recorder, WTSIntentDriftGuard, WTSToolOutputGuard, build_pipeline

    out = REPO / "results/e3"
    (out / "agentdojo_guard_log").mkdir(parents=True, exist_ok=True)
    ac = cfg["agentdojo"]
    enc = build_encoder(cfg["encoder"])
    refuse_test_only(enc)
    bv = ac["benchmark_version"]
    for lc in ac["llms"]:
        for seed in cfg["seeds"]:
            llm = make_llm(lc)
            # 1) calibration: benign runs on every suite's calibration tasks, pooled into one guard
            recd = Recorder()
            p_cal = build_pipeline(llm, f"{llm.name}-calib", extra_after_tools=[recd], extra_after_llm=[recd])
            splits = {}
            for sname in ac["suites"]:
                suite = get_suite(bv, sname)
                cal_ids, test_ids = split_tasks(suite, ac["calibration_task_frac"], seed)
                splits[sname] = (suite, test_ids)
                benchmark_suite_without_injections(p_cal, suite, None, True, user_tasks=cal_ids, benchmark_version=bv)
            outputs = {t: [enc.encode(x) for x in v] for t, v in recd.outputs.items()}
            guard = AgentGuard(alpha=cfg["alpha"], min_tool_ref_units=10, ref_kwargs=dict(seed=seed))
            guard.fit_outputs(outputs, seed=seed)
            guard.fit_drift([(enc.encode(a), enc.encode(b)) for a, b in recd.plan_pairs], seed=seed)

            for sname, (suite, test_ids) in splits.items():
                with run_record("e3", dict(cfg, mode="agents", llm=lc["model"], suite=sname), seed) as rec:
                    rows = []
                    for dfn in ac["defenses"]:
                        olog, dlog = [], []
                        if dfn in ("none", "spotlighting_with_delimiting", "transformers_pi_detector"):
                            pc = PipelineConfig(llm=llm, model_id=None, defense=None if dfn == "none" else dfn,
                                                system_message_name=None, system_message=None)
                            pipe = AgentPipeline.from_config(pc)
                        else:
                            n = 2 if dfn == "wts_full" else 1
                            after_tools, after_llm = [], []
                            if dfn in ("wts_output_block", "wts_output_quarantine", "wts_full"):
                                act = "quarantine" if dfn == "wts_output_quarantine" else "block"
                                after_tools.append(WTSToolOutputGuard(guard, enc, act, n_tests=n, log=olog))
                            if dfn in ("wts_drift_confirm", "wts_full"):
                                after_llm.append(WTSIntentDriftGuard(guard, enc, n_tests=n, log=dlog))
                            pipe = build_pipeline(llm, f"{llm.name}-{dfn}", after_tools, after_llm)
                        res_benign = benchmark_suite_without_injections(pipe, suite, None, True, user_tasks=test_ids,
                                                                        benchmark_version=bv)
                        for (uid, _), u in res_benign["utility_results"].items():
                            rows.append(dict(llm=lc["model"], seed=seed, suite=sname, defense=dfn, attack="none",
                                             user_task=uid, injection_task="", utility=u, security=None))
                        for aname in ac["attacks"]:
                            attack = load_attack(aname, suite, pipe)
                            res = benchmark_suite_with_injections(pipe, suite, attack, None, True, user_tasks=test_ids,
                                                                  benchmark_version=bv, verbose=False)
                            for key, u in res["utility_results"].items():
                                rows.append(dict(llm=lc["model"], seed=seed, suite=sname, defense=dfn, attack=aname,
                                                 user_task=key[0], injection_task=key[1], utility=u,
                                                 security=res["security_results"][key]))
                        tag = f"{lc['model'].split('/')[-1]}_{sname}_{dfn}_seed{seed}"
                        if olog:
                            pd.DataFrame(olog).to_csv(out / "agentdojo_guard_log" / f"out_{tag}.csv.gz", index=False)
                        if dlog:
                            pd.DataFrame(dlog).to_csv(out / "agentdojo_guard_log" / f"drift_{tag}.csv.gz", index=False)
                    append_rows(out / "agentdojo_items.csv", rows)
                    rec.update(n_rows=len(rows), n_cal_outputs=sum(len(v) for v in outputs.values()),
                               n_plan_pairs=len(recd.plan_pairs), pooled_can_flag=guard.pooled_can_flag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/e3.yaml"))
    ap.add_argument("--gt-detection", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.smoke or a.gt_detection:
        run_gt_detection(cfg, smoke=a.smoke)
    else:
        run_agents(cfg)
