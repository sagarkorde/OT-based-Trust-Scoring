"""Phase 2: reproduction gates G1-G4 (reproduction.md). No defense anywhere; each benchmark's own protocol.

  G1  PoisonedRAG black-box, NQ, Contriever, top-5, 5 adversarial passages -> ASR
  G2  AgentDojo important_instructions, all suites, all user tasks, no defense -> utility, targeted ASR
  G3  InjecAgent base setting, prompted agent, DH + DS -> ASR-valid (InjecAgent's get_score definition)
  G4  HarmBench DirectRequest on the standard text behaviors, HarmBench classifier -> ASR

Writes results/gates/<gate>.csv (per item) and results/gates/summary.csv; prints the numbers to copy into
reproduction.md next to the published value read from the cited table.

  python tasks.py gates --profile laptop [-- --only G1 G4]
"""
from __future__ import annotations

import argparse
import json
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
from wts.runlog import append_rows, run_record  # noqa: E402

OUT = REPO / "results/gates"


def _out():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def progress(gate, i, n, t0, every=25):
    """One line every `every` items: done/total, elapsed, estimated time left."""
    if (i + 1) % every and i + 1 != n:
        return
    el = time.time() - t0
    left = el / (i + 1) * (n - i - 1)
    print(f"[{gate}] {i + 1}/{n}  elapsed {el / 60:.1f} min  ~{left / 60:.0f} min left", flush=True)


def summary(gate, **kw):
    """One JSON object per line: gates report different fields, which a single CSV header cannot hold."""
    row = dict(gate=gate, date=time.strftime("%Y-%m-%d %H:%M"), **kw)
    with (_out() / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"\n[{gate}] " + "  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def g1(c):
    import run_e2_rag as E2
    corpus, _, _ = D.beir_corpus(c["dataset"])
    targets = D.poisonedrag_targets(c["dataset"])[: c.get("n_targets", 100)]
    retrieval = D.poisonedrag_retrieval(c["dataset"], c["retriever"])
    R = E2.Retriever(c["retriever"])
    llm = build_llm(c["llm"])
    rows, t0 = [], time.time()
    for k, t in enumerate(targets):
        progress("G1", k, len(targets), t0, every=10)
        pool = E2.build_pool(t, retrieval, corpus, R, D.poisonedrag_attack_texts(t, c["adv_per_query"]), c["top_k"])
        resp = llm.ask(E2.wrap_prompt(t["question"], [p[0] for p in pool[: c["top_k"]]]))
        rows.append(dict(qid=t["id"], n_adv=sum(p[1] for p in pool[: c["top_k"]]),
                         success=E2.clean_str(t["incorrect"]) in E2.clean_str(resp), response=resp[:300]))
    df = pd.DataFrame(rows)
    df.to_csv(_out() / "G1.csv", index=False)
    free_gpu(R)
    summary("G1", llm=c["llm"]["model"], n=len(df), asr=round(df.success.mean(), 4),
            mean_adv_in_top_k=round(df.n_adv.mean(), 2))


def g2(c):
    from agentdojo.attacks.attack_registry import load_attack
    from agentdojo.benchmark import benchmark_suite_with_injections, benchmark_suite_without_injections
    from agentdojo.task_suite.load_suites import get_suite

    from run_e3_agentdojo import make_llm
    from wts.agentdojo_hook import build_pipeline
    llm = make_llm(c["llm"])
    pipe = build_pipeline(llm, f"{llm.name}-nodefense")
    rows = []
    for sname in c["suites"]:
        suite = get_suite(c["benchmark_version"], sname)
        ben = benchmark_suite_without_injections(pipe, suite, None, True, benchmark_version=c["benchmark_version"])
        rows += [dict(suite=sname, attack="none", user_task=k[0], injection_task="", utility=v, security=None)
                 for k, v in ben["utility_results"].items()]
        res = benchmark_suite_with_injections(pipe, suite, load_attack(c["attack"], suite, pipe), None, True,
                                              benchmark_version=c["benchmark_version"], verbose=False)
        rows += [dict(suite=sname, attack=c["attack"], user_task=k[0], injection_task=k[1], utility=v,
                      security=res["security_results"][k]) for k, v in res["utility_results"].items()]
        pd.DataFrame(rows).to_csv(_out() / "G2.csv", index=False)
    df = pd.DataFrame(rows)
    a = df[df.attack == c["attack"]]
    summary("G2", llm=c["llm"]["model"], utility=round(df[df.attack == "none"].utility.mean(), 4),
            utility_under_attack=round(a.utility.mean(), 4), targeted_asr=round(a.security.astype(float).mean(), 4),
            n_cases=len(a))


def g3(c):
    import run_e3_injecagent as IA
    llm = build_llm(c["llm"])
    rows = []
    with IA.in_injecagent():
        from src.output_parsing import evaluate_output_prompted
        from src.prompts.agent_prompts import PROMPT_DICT
        from src.utils import get_tool_dict
        tool_dict = get_tool_dict()
        sys_p, user_p = PROMPT_DICT["InjecAgent"]
        cases = [(split, i, item) for split in ("dh", "ds") for i, item in enumerate(D.injecagent_cases(split, c["setting"]))]
        t0 = time.time()
        for k, (split, i, item) in enumerate(cases):
            progress("G3", k, len(cases), t0)
            if True:
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
                rows.append(dict(split=split, case=i, eval=res, sim_response=sim_src))
    df = pd.DataFrame(rows)
    df.to_csv(_out() / "G3.csv", index=False)
    out = {}
    for split in ("dh", "ds"):
        v = df[(df.split == split) & df["eval"].isin(["succ", "unsucc"])]
        out[f"asr_valid_{split}"] = round((v["eval"] == "succ").mean(), 4) if len(v) else None
    v = df[df["eval"].isin(["succ", "unsucc"])]
    summary("G3", llm=c["llm"]["model"], setting=c["setting"], asr_valid_total=round((v["eval"] == "succ").mean(), 4),
            valid_rate=round(len(v) / len(df), 4), ds_step2_local_simulations=int(df.sim_response.str.startswith("local").sum()),
            **out)


def g4(c):
    from run_e4_adaptive import _harmbench_judge
    llm = build_llm(c["llm"])
    behaviors = D.harmbench_direct_requests()
    gens, t0 = [], time.time()
    for k, b in enumerate(behaviors):         # 1) generations (GPU, Ollama)
        progress("G4-generate", k, len(behaviors), t0)
        gens.append(dict(behavior_id=b["behavior_id"], behavior=b["text"],
                         generation=llm.ask(b["text"], max_tokens=c.get("max_new_tokens", 512))))
    pd.DataFrame(gens).to_csv(_out() / "G4_generations.csv", index=False)
    ollama_unload_all()
    judge = _harmbench_judge(c["judge"])     # 2) HarmBench classifier, bf16 (CPU on the laptop profile)
    t0 = time.time()
    for k, g in enumerate(gens):
        progress("G4-judge", k, len(gens), t0, every=10)
        g["jailbroken"] = judge(g["behavior"], g["generation"])
    df = pd.DataFrame(gens)
    df.to_csv(_out() / "G4.csv", index=False)
    summary("G4", llm=c["llm"]["model"], method="DirectRequest", n=len(df), asr=round(df.jailbroken.mean(), 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/laptop/gates.yaml"))
    ap.add_argument("--only", nargs="*", default=["G1", "G2", "G3", "G4"])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    for g in a.only:
        with run_record("gates", dict(cfg[g], gate=g), 0) as rec:
            {"G1": g1, "G2": g2, "G3": g3, "G4": g4}[g](cfg[g])
