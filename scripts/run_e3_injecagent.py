"""E3 (InjecAgent): WTS on the user-tool response, scored with InjecAgent's own prompt and evaluator (RQ3).

Detection (LLM-free): attacked 'Tool Response' vs the same response template with the attacker slot filled
by benign text. Calibration uses benign fillers disjoint from those used for the benign test responses.
Agent mode: InjecAgent's predict_one_case / evaluate_output_prompted unchanged; with the WTS defense, a
flagged tool response is replaced by AgentDojo's redaction string before the model sees it. ASR-valid =
succ / (succ + unsucc), as in InjecAgent's get_score.

--smoke: detection only, test-only hashing encoder, local benign fillers; results/_smoke/.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wts import data as D  # noqa: E402
from wts.encoders import HashingEncoder, build_encoder  # noqa: E402
from wts.guards import AgentGuard  # noqa: E402
from wts.runlog import append_rows, refuse_test_only, run_record  # noqa: E402
from wts.stats import detection_metrics  # noqa: E402

IA = REPO / "ext/InjecAgent"
REDACTED = "<Data omitted because a prompt injection was detected>"


@contextmanager
def in_injecagent():
    old = os.getcwd()
    os.chdir(IA)
    sys.path.insert(0, str(IA))
    try:
        yield
    finally:
        os.chdir(old)


def fillers(smoke: bool, n: int, seed: int):
    if smoke:
        pool = [t["question"] for ds in ("nq", "hotpotqa", "msmarco") for t in D.poisonedrag_targets(ds)]
    else:
        pool = [r["text"] for r in D.alpaca_prompts(4 * n, 7000 + seed)]
    random.Random(seed).shuffle(pool)
    return pool


def build_guard(cases, enc, fill, alpha, seed):
    """Per user-tool reference + calibration from benign responses (slot filled with benign text)."""
    by_tool = {}
    for i, c in enumerate(cases):
        for j in range(2):
            by_tool.setdefault(c["User Tool"], []).append(
                enc.encode(D.injecagent_benign_response(c, fill[(2 * i + j) % len(fill)])))
    g = AgentGuard(alpha=alpha, min_tool_ref_units=10, use_drift=False, ref_kwargs=dict(seed=seed))
    return g.fit_outputs(by_tool, seed=seed)


def run(cfg, smoke=False):
    ic = cfg["injecagent"]
    out = REPO / ("results/_smoke/e3" if smoke else "results/e3")
    enc = HashingEncoder(dim=64) if smoke else build_encoder(cfg["encoder"])
    if not smoke:
        refuse_test_only(enc)
    seeds = cfg["seeds"][:1] if smoke else cfg["seeds"]
    for seed in seeds:
        for setting in ic["settings"]:
            cases = {s: D.injecagent_cases(s, setting) for s in ic["splits"]}
            allc = [c for s in ic["splits"] for c in cases[s]]
            # unique templates: references and calibration use fillers [0, N); benign test uses [N, 2N)
            uniq = list({c["Tool Response Template"]: c for c in allc}.values())
            fill = fillers(smoke, 200 * len(uniq) + 4000, seed)
            half = len(fill) // 2
            with run_record("_smoke/e3" if smoke else "e3", dict(cfg, mode="injecagent", setting=setting,
                                                                 encoder=enc.name), seed, REPO / "results") as rec:
                guard = build_guard(uniq * 50, enc, fill[:half], cfg["alpha"], seed)  # 100 benign responses per template
                # detection: every attacked response vs one benign response per case (held-out fillers)
                rows = []
                for s in ic["splits"]:
                    for i, c in enumerate(cases[s]):
                        sa, pa = guard.output_pvalue(c["User Tool"], enc.encode(c["Tool Response"]))
                        b = D.injecagent_benign_response(c, fill[half + i % (len(fill) - half)])
                        sb, pb = guard.output_pvalue(c["User Tool"], enc.encode(b))
                        rows += [dict(seed=seed, setting=setting, split=s, case=i, injected=True, score=sa, p=pa),
                                 dict(seed=seed, setting=setting, split=s, case=i, injected=False, score=sb, p=pb)]
                y = np.array([r["injected"] for r in rows], int)
                sc = np.array([r["score"] for r in rows])
                fl = np.array([r["p"] <= cfg["alpha"] for r in rows])
                det = dict(seed=seed, setting=setting, encoder=enc.name, **detection_metrics(y, sc),
                           tpr_at_alpha=float(fl[y == 1].mean()), fpr_at_alpha=float(fl[y == 0].mean()))
                append_rows(out / "injecagent_detection_items.csv", rows)
                append_rows(out / "injecagent_detection.csv", [det])
                print(f"[e3-ia] seed={seed} {setting}: AUROC={det['auroc']:.3f} TPR@a={det['tpr_at_alpha']:.2f} "
                      f"FPR@a={det['fpr_at_alpha']:.2f} (n={len(rows)})", flush=True)
                if not smoke:
                    run_agents(cfg, cases, guard, enc, seed, setting, out)
                rec.update(detection=det)


def run_agents(cfg, cases, guard, enc, seed, setting, out):
    from wts.llm import build_llm
    ic = cfg["injecagent"]
    with in_injecagent():
        from src.output_parsing import evaluate_output_prompted
        from src.prompts.agent_prompts import PROMPT_DICT
        from src.utils import get_simulated_attacker_tool_response, get_tool_dict
        tool_dict = get_tool_dict()
        sys_p, user_p = PROMPT_DICT["InjecAgent"]
        for lc in ic["llms"]:
            llm = build_llm(lc)
            for dfn in ic["defenses"]:
                rows = []
                for split, cs in cases.items():
                    for i, c in enumerate(cs):
                        item = dict(c)
                        flagged = False
                        if dfn == "wts_output_block":
                            _, p = guard.output_pvalue(c["User Tool"], enc.encode(c["Tool Response"]))
                            flagged = p <= cfg["alpha"]
                            if flagged:
                                item["Tool Response"] = REDACTED
                        out1 = _predict(sys_p, user_p, item, tool_dict, llm, step_2=False)
                        ev = evaluate_output_prompted(out1, item["Attacker Tools"][0], item["Attacker Tools"],
                                                      item["User Tool"], item["Attacker Instruction"])
                        res = ev["eval"]
                        if split == "ds" and res == "succ":
                            item["Attacker Tool Parameters"] = ev["Attacker Tool Parameters"]
                            item["Attacker Tool Thought"] = ev["Attacker Tool Thought"]
                            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(
                                item["Attacker Tools"][0], item["Attacker Tool Parameters"], tool_dict)
                            out2 = _predict(sys_p, user_p, item, tool_dict, llm, step_2=True)
                            res = evaluate_output_prompted(out2, item["Attacker Tools"][1], item["Attacker Tools"],
                                                           item["User Tool"], item["Attacker Instruction"])["eval"]
                        rows.append(dict(llm=lc["model"], seed=seed, setting=setting, defense=dfn, split=split,
                                         case=i, flagged=flagged, eval=res))
                append_rows(out / "injecagent_items.csv", rows)


def _predict(sys_p, user_p, item, tool_dict, llm, step_2):
    """Same prompt construction as InjecAgent's predict_one_case, sent through our OpenAI-compatible client."""
    names = [item["User Tool"]] + item["Attacker Tools"]
    desc = json.dumps([tool_dict[t] for t in names], indent=True)
    if step_2:
        pad = (f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\n"
               f"Observation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\n"
               f"Action: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\n"
               f"Observation: {item['Attacker Tool Response']}\n")
    else:
        pad = (f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\n"
               f"Observation: {item['Tool Response']}\n")
    up = user_p.format(available_tool_descriptions=desc, input=item["User Instruction"],
                       available_tool_names=names, agent_scratchpad=pad)
    return llm.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": up}])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/e3.yaml"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run(yaml.safe_load(open(a.config)), smoke=a.smoke)
