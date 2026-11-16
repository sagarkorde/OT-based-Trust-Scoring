"""Phase 2: reproduction gates G1-G4 (reproduction.md). No defense anywhere; each benchmark's own protocol.

  G1  PoisonedRAG black-box, NQ, Contriever, top-5, 5 adversarial passages -> ASR
  G2  AgentDojo important_instructions, all suites, all user tasks, no defense -> utility, targeted ASR
  G3  InjecAgent base setting, prompted agent, DH + DS -> ASR-valid (InjecAgent's get_score definition)
  G4  HarmBench DirectRequest on the standard text behaviors, HarmBench classifier -> ASR

Resumable: every finished item is journaled under results/gates/<gate>/<tag>.jsonl (tag = config + gate version),
so re-running the same command continues where it stopped. GATE_VERSION is bumped whenever a gate's logic changes,
which starts a fresh journal (G1 is at 2 after the question-prefix fix). AgentDojo (G2) resumes through its own
per-task logs in results/agentdojo_logs/.

Outputs: results/gates/<gate>.csv (per item) and results/gates/summary.jsonl (one line per finished gate).

  python tasks.py gates --profile laptop --only G1 G4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from wts import data as D  # noqa: E402
from wts.hw import free_gpu, ollama_unload_all  # noqa: E402
from wts.llm import build_llm  # noqa: E402
from wts.resume import Journal, Progress, tag_of  # noqa: E402
from wts.runlog import run_record  # noqa: E402

OUT = REPO / "results/gates"
GATE_VERSION = {"G1": 2, "G2": 1, "G3": 1, "G3_v01": 1, "G4": 1}


def _out():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def journal(gate, c, part=""):
    tag = tag_of(c, GATE_VERSION[gate])
    return Journal(_out() / gate / f"{part}{tag}.jsonl")


def summary(gate, **kw):
    """One JSON object per line: gates report different fields, which a single CSV header cannot hold."""
    row = dict(gate=gate, date=time.strftime("%Y-%m-%d %H:%M"), version=GATE_VERSION[gate], **kw)
    with (_out() / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"\n[{gate}] RESULT  " + "  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


# ---------------------------------------------------------------------------------------------- G1
def g1(c):
    import run_e2_rag as E2
    J = journal("G1", c)
    targets = D.poisonedrag_targets(c["dataset"])[: c.get("n_targets", 100)]
    todo = [t for t in targets if t["id"] not in J]
    if todo:
        corpus, _, _ = D.beir_corpus(c["dataset"])
        retrieval = D.poisonedrag_retrieval(c["dataset"], c["retriever"])
        R = E2.Retriever(c["retriever"])
        llm = build_llm(c["llm"])
        P = Progress("G1 PoisonedRAG", len(targets), resumed=len(targets) - len(todo))
        for t in todo:
            pool = E2.build_pool(t, retrieval, corpus, R, D.poisonedrag_attack_texts(t, c["adv_per_query"]), c["top_k"])
            resp = llm.ask(E2.wrap_prompt(t["question"], [p[0] for p in pool[: c["top_k"]]]))
            J.add(t["id"], dict(qid=t["id"], n_adv=sum(p[1] for p in pool[: c["top_k"]]),
                                success=E2.clean_str(t["incorrect"]) in E2.clean_str(resp), response=resp[:300]))
            P.step()
        free_gpu(R)
    df = pd.DataFrame([J.get(t["id"]) for t in targets]).drop(columns="_key")
    df.to_csv(_out() / "G1.csv", index=False)
    summary("G1", llm=c["llm"]["model"], n=len(df), asr=round(df.success.mean(), 4),
            mean_adv_in_top_k=round(df.n_adv.mean(), 2))


# ---------------------------------------------------------------------------------------------- G2
def g2(c):
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
    from agentdojo.task_suite.load_suites import get_suite

    from run_e3_agentdojo import ProgressElement, count_logged, make_llm, reset_for_attack
    from wts.agentdojo_hook import QuietLogger, build_pipeline
    logdir = REPO / "results/agentdojo_logs"
    try:
        with QuietLogger(logdir):      # AgentDojo saves each task's trajectory only inside a logger context
            _g2(c, logdir, ProgressElement, count_logged, make_llm, build_pipeline, load_attack, get_suite,
                benchmark_suite_with_injections, benchmark_suite_without_injections)
    except Exception as e:             # account problems: say what to fix instead of a long traceback
        msg = str(e)
        if "no credits" in msg or "insufficient_quota" in msg:
            sys.exit("[G2] stopped: the OpenAI API organization of this key has no credits. Claim the ChatGPT Pro API "
                     "credits for that organization (or add credit) at platform.openai.com > Settings > Billing, then "
                     "re-run. Nothing finished is lost; the interrupted task is redone.")
        if "invalid_api_key" in msg or "Incorrect API key" in msg:
            sys.exit("[G2] stopped: OpenAI rejected the API key. Create a new key and set OPENAI_API_KEY again.")
        raise


def _g2(c, logdir, ProgressElement, count_logged, make_llm, build_pipeline, load_attack, get_suite,
        benchmark_suite_with_injections, benchmark_suite_without_injections):
    if c["llm"].get("provider") == "openai" and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("G2 needs an OpenAI API key: run  setx OPENAI_API_KEY \"<your key>\"  in PowerShell, close the window, "
                 "open a new one and run the gate again. (Never paste the key into a chat or a file in the repo.)")
    llm = make_llm(c["llm"])
    print(f"[G2] model {c['llm']['model']}, AgentDojo benchmark {c['benchmark_version']}, attack {c['attack']}",
          flush=True)
    tag = tag_of(c, GATE_VERSION["G2"])
    rows = []
    for sname in c["suites"]:
        suite = get_suite(c["benchmark_version"], sname)
        attack_name = c["attack"]
        pe = ProgressElement()
        pipe = build_pipeline(llm, f"{llm.name}-gate-{tag}", first=[pe])
        pe.reset(f"G2 {sname} benign", len(suite.user_tasks), count_logged(logdir, pipe.name, sname, "none"))
        ben = benchmark_suite_without_injections(pipe, suite, logdir, False, benchmark_version=c["benchmark_version"])
        rows += [dict(suite=sname, attack="none", user_task=k[0], injection_task="", utility=v, security=None)
                 for k, v in ben["utility_results"].items()]
        attack = load_attack(attack_name, suite, pipe)
        reset_for_attack(pe, f"G2 {sname} {attack_name}", logdir, pipe.name, suite, attack, attack_name,
                         len(suite.user_tasks))
        res = benchmark_suite_with_injections(pipe, suite, attack, logdir, False,
                                              benchmark_version=c["benchmark_version"], verbose=False)
        rows += [dict(suite=sname, attack=attack_name, user_task=k[0], injection_task=k[1], utility=v,
                      security=res["security_results"][k]) for k, v in res["utility_results"].items()]
        pd.DataFrame(rows).to_csv(_out() / "G2.csv", index=False)
    df = pd.DataFrame(rows)
    a = df[df.attack == c["attack"]]
    summary("G2", llm=c["llm"]["model"], utility=round(df[df.attack == "none"].utility.mean(), 4),
            utility_under_attack=round(a.utility.mean(), 4), targeted_asr=round(a.security.astype(float).mean(), 4),
            n_cases=len(a))


# ---------------------------------------------------------------------------------------------- G3
def g3(c, gate="G3"):
    """InjecAgent ASR-valid. gate="G3_v01": the same run with Mistral-7B-Instruct-v0.1 (sensitivity check for the
    unnamed Mistral release in InjecAgent), written to its own journal, CSV and summary line."""
    import run_e3_injecagent as IA
    J = journal(gate, c)
    legacy = _out() / "G3.csv"                      # finished before journals existed (same config, v1)
    if gate == "G3" and len(J) == 0 and legacy.exists():
        old = pd.read_csv(legacy).fillna("")
        for r in old.itertuples():
            J.add(Journal.key(r.split, r.case), dict(split=r.split, case=int(r.case), eval=r.eval,
                                                    sim_response=r.sim_response, imported_from="G3.csv"))
        print(f"[G3] imported {len(old)} finished cases from the earlier run", flush=True)
    llm = build_llm(c["llm"])
    with IA.in_injecagent():
        from src.output_parsing import evaluate_output_prompted
        from src.prompts.agent_prompts import PROMPT_DICT
        from src.utils import get_tool_dict
        tool_dict = get_tool_dict()
        sys_p, user_p = PROMPT_DICT["InjecAgent"]
        cases = [(split, i, item) for split in ("dh", "ds") for i, item in enumerate(D.injecagent_cases(split, c["setting"]))]
        todo = [x for x in cases if Journal.key(x[0], x[1]) not in J]
        P = Progress(f"{gate} InjecAgent", len(cases), resumed=len(cases) - len(todo))
        for split, i, item in todo:
            item = dict(item)
            o1 = IA._predict(sys_p, user_p, item, tool_dict, llm, step_2=False)
            ev = evaluate_output_prompted(o1, item["Attacker Tools"][0], item["Attacker Tools"], item["User Tool"],
                                          item["Attacker Instruction"])
            res, sim_src = ev["eval"], ""
            if split == "ds" and res == "succ":
                item.update({k: ev[k] for k in ("Attacker Tool Parameters", "Attacker Tool Thought")})
                item["Attacker Tool Response"], sim_src = IA.simulated_attacker_response(
                    item["Attacker Tools"][0], item["Attacker Tool Parameters"], tool_dict, llm)
                o2 = IA._predict(sys_p, user_p, item, tool_dict, llm, step_2=True)
                res = evaluate_output_prompted(o2, item["Attacker Tools"][1], item["Attacker Tools"],
                                               item["User Tool"], item["Attacker Instruction"])["eval"]
            J.add(Journal.key(split, i), dict(split=split, case=i, eval=res, sim_response=sim_src))
            P.step()
    df = pd.DataFrame([J.get(Journal.key(s, i)) for s, i, _ in cases]).drop(columns="_key")
    df["sim_response"] = df.sim_response.fillna("")
    df.to_csv(_out() / f"{gate}.csv", index=False)
    out = {}
    for split in ("dh", "ds"):
        v = df[(df.split == split) & df["eval"].isin(["succ", "unsucc"])]
        out[f"asr_valid_{split}"] = round((v["eval"] == "succ").mean(), 4) if len(v) else None
    v = df[df["eval"].isin(["succ", "unsucc"])]
    summary(gate, llm=c["llm"]["model"], weights=c["llm"].get("ollama_base"), setting=c["setting"], asr_valid_total=round((v["eval"] == "succ").mean(), 4),
            valid_rate=round(len(v) / len(df), 4),
            ds_step2_local_simulations=int(df.sim_response.str.startswith("local").sum()), **out)


# ---------------------------------------------------------------------------------------------- G4
def g4(c):
    from run_e4_adaptive import _harmbench_judge
    Jg, Jj = journal("G4", c, "generate_"), journal("G4", c, "judge_")
    behaviors = D.harmbench_direct_requests()
    legacy = _out() / "G4_generations.csv"         # generations written before journals existed (same config, v1)
    if len(Jg) == 0 and legacy.exists():
        old = pd.read_csv(legacy).fillna("")
        for r in old.itertuples():
            Jg.add(r.behavior_id, dict(behavior_id=r.behavior_id, behavior=r.behavior, generation=r.generation,
                                       imported_from="G4_generations.csv"))
        print(f"[G4] imported {len(old)} generations saved by the earlier run", flush=True)
    todo = [b for b in behaviors if b["behavior_id"] not in Jg]
    if todo:                                        # 1) generations (GPU, Ollama)
        llm = build_llm(c["llm"])
        P = Progress("G4 generate", len(behaviors), resumed=len(behaviors) - len(todo))
        for b in todo:
            Jg.add(b["behavior_id"], dict(behavior_id=b["behavior_id"], behavior=b["text"],
                                          generation=llm.ask(b["text"], max_tokens=c.get("max_new_tokens", 512))))
            P.step()
    todo = [b for b in behaviors if b["behavior_id"] not in Jj]
    if todo:                                        # 2) HarmBench classifier, bf16 (CPU on the laptop profile)
        ollama_unload_all()
        judge = _harmbench_judge(c["judge"])
        P = Progress("G4 judge", len(behaviors), resumed=len(behaviors) - len(todo), every_s=60)
        for b in todo:
            g = Jg.get(b["behavior_id"])
            Jj.add(b["behavior_id"], dict(behavior_id=b["behavior_id"], jailbroken=bool(judge(g["behavior"], g["generation"]))))
            P.step()
    df = pd.DataFrame([{**Jg.get(b["behavior_id"]), "jailbroken": Jj.get(b["behavior_id"])["jailbroken"]}
                       for b in behaviors]).drop(columns="_key")
    df.to_csv(_out() / "G4.csv", index=False)
    summary("G4", llm=c["llm"]["model"], method="DirectRequest", n=len(df), asr=round(df.jailbroken.mean(), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/laptop/gates.yaml"))
    ap.add_argument("--only", nargs="*", default=["G1", "G2", "G3", "G4"])   # G3_v01: sensitivity check, on request
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    for g in a.only:
        print(f"\n=== {g} ===", flush=True)
        with run_record("gates", dict(cfg[g], gate=g, version=GATE_VERSION[g]), 0) as rec:
            {"G1": g1, "G2": g2, "G3": g3, "G3_v01": lambda c: g3(c, "G3_v01"), "G4": g4}[g](cfg[g])
