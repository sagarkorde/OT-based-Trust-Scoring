"""E1: prompt-channel detection of injections and jailbreaks (RQ1), calibration study (RQ4), judge timing (RQ6).

Outputs (all read by make_tables.py / make_figures.py; nothing is typed into the paper):
  results/e1/scores/<encoder>__seed<k>.csv.gz   per-item scores for every measure (paired tests, CIs)
  results/e1/scores/baselines__seed<k>.csv.gz   per-item scores for LLM baselines
  results/e1/detection.csv                      point metrics per encoder x measure x family x seed
  results/e1/decisions.csv                      TPR / FPR at the conformal threshold (alpha)
  results/e1/calibration.csv                    RQ4 coverage study, in distribution and under shift
  results/e5_overhead.csv                       encoding + scoring + judge latency on real prompts

--smoke runs the whole pipeline with the test-only hashing encoder on local benchmark text, no downloads,
and writes under results/_smoke/. Smoke numbers are plumbing checks, never results.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wts import data as D  # noqa: E402
from wts.cache import encode_cached  # noqa: E402
from wts.conformal import ConformalCalibrator, coverage_study  # noqa: E402
from wts.encoders import HashingEncoder, PCAProjector, build_encoder  # noqa: E402
from wts.hw import free_gpu, ollama_unload_all  # noqa: E402
from wts.measures import ReferenceModel  # noqa: E402
from wts.resume import Progress, mark_done, purge_rows, resumable_map, tag_of  # noqa: E402
from wts.runlog import append_rows, refuse_test_only, run_record  # noqa: E402

E1_VERSION = 1   # bump when the scoring code changes: finished (encoder, seed) markers of older versions are ignored
from wts.stats import detection_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------ data
def load_items(cfg, seed_for_sampling=1234):
    b, a = cfg["benign"], cfg["attacks"]
    alpaca = D.alpaca_prompts(b["alpaca_n"] + b.get("pca_fit_n", 0), seed_for_sampling, b["revisions"].get("alpaca"))
    pca_pool, alpaca = alpaca[b["alpaca_n"]:], alpaca[: b["alpaca_n"]]
    lmsys = D.lmsys_prompts(b["lmsys_n"], seed_for_sampling, b["revisions"].get("lmsys"))
    attacks = []
    if a.get("deepset"):
        attacks += D.deepset_injections(a["revisions"].get("deepset"))
    if a.get("in_the_wild_n"):
        attacks += D.in_the_wild_jailbreaks(a["in_the_wild_n"], seed_for_sampling, revision=a["revisions"].get("in_the_wild"))
    if a.get("harmbench_direct"):
        attacks += D.harmbench_direct_requests()
    for method, path in (a.get("harmbench_test_cases") or {}).items():
        attacks += D.harmbench_test_cases(REPO / path, method)
    return alpaca, lmsys, attacks, pca_pool


def load_smoke_items():
    """Local text only: PoisonedRAG target questions as benign prompts; InjecAgent attacker instructions and
    HarmBench direct requests as attacks. Plumbing check only."""
    qs = []
    for ds in ("nq", "hotpotqa", "msmarco"):
        qs += [dict(id=f"q-{ds}-{t['id']}", text=t["question"], source=ds, label=0, family="benign")
               for t in D.poisonedrag_targets(ds)]
    random.Random(0).shuffle(qs)
    alpaca, lmsys = qs[:200], qs[200:300]
    inst = sorted({c["Attacker Instruction"] for s in ("dh", "ds") for c in D.injecagent_cases(s)})
    attacks = [dict(id=f"ia-{i}", text=t, source="injecagent", label=1, family="prompt_injection")
               for i, t in enumerate(inst)]
    attacks += D.harmbench_direct_requests()[:60]
    return alpaca, lmsys, attacks, []


# ------------------------------------------------------------------------------------------ main
def run(cfg, smoke=False):
    out = REPO / ("results/_smoke/e1" if smoke else "results/e1")
    (out / "scores").mkdir(parents=True, exist_ok=True)
    measures = cfg["measures"]
    if smoke:
        alpaca, lmsys, attacks, pca_pool = load_smoke_items()
        enc_cfgs = [dict(kind="hashing", dim=64)]
        split = dict(ref=60, cal=100, test=40)
        cfg = {**cfg, "calibration_study": {**cfg["calibration_study"], "n_cals": [50, 100], "n_repeats": 50}}
        seeds = cfg["seeds"][:2]
    else:
        alpaca, lmsys, attacks, pca_pool = load_items(cfg)
        ollama_unload_all()

        def _make(ec):  # built lazily, one at a time, so only one model occupies the GPU
            e = build_encoder(ec)
            refuse_test_only(e)
            if ec["kind"] == "llm_layer":
                proj = PCAProjector(n_components=ec.get("pca_dim", 768))
                proj.fit(e.raw_many([r["text"] for r in pca_pool]))
                e.projector = proj
            return e
        enc_cfgs = cfg["encoders"]
        split = cfg["benign"]["split"]
        seeds = cfg["seeds"]

    families = sorted({r["family"] for r in attacks})
    run_cfg = {k: v for k, v in cfg.items() if k not in ("encoders", "seeds", "baselines")}

    def marker(ec, seed):
        return out / "done" / f"{tag_of(dict(run_cfg, encoder=ec), E1_VERSION)}_seed{seed}.ok"

    P = Progress("E1 encoders x seeds", len(enc_cfgs) * len(seeds),
                 resumed=sum(marker(ec, sd).exists() for ec in enc_cfgs for sd in seeds), every_s=0)
    for ec in enc_cfgs:
        todo_seeds = [sd for sd in seeds if smoke or not marker(ec, sd).exists()]
        if not todo_seeds:
            print(f"[E1] encoder {ec.get('name', ec.get('model', ec['kind']))}: all seeds done, skipping", flush=True)
            continue
        enc = HashingEncoder(dim=64) if smoke else _make(ec)
        t0 = time.perf_counter()
        U_alp = encode_cached(enc, [r["text"] for r in alpaca])
        U_lms = encode_cached(enc, [r["text"] for r in lmsys])
        U_att = encode_cached(enc, [r["text"] for r in attacks])
        enc_ms = 1000 * (time.perf_counter() - t0) / max(1, len(alpaca) + len(lmsys) + len(attacks))

        for seed in todo_seeds:
            for f in ("detection.csv", "decisions.csv", "calibration.csv"):   # drop a partly written seed
                purge_rows(out / f, {"encoder": enc.name, "seed": seed})
            with run_record("e1" if not smoke else "_smoke/e1", dict(cfg, encoder=enc.name), seed, REPO / "results") as rec:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(len(alpaca))
                i_ref = perm[: split["ref"]]
                i_cal = perm[split["ref"]: split["ref"] + split["cal"]]
                i_te = perm[split["ref"] + split["cal"]: split["ref"] + split["cal"] + split["test"]]
                ref = ReferenceModel(seed=seed).fit([U_alp[i] for i in i_ref])

                def score(units, label=""):
                    Ps = Progress(f"E1 {enc.name} seed {seed} scoring {label}", len(units), every_s=60) \
                        if len(units) > 500 else None
                    res = []
                    for X in units:
                        res.append(ref.score_all(X, measures))
                        if Ps:
                            Ps.step()
                    return res

                t1 = time.perf_counter()
                S_cal = score([U_alp[i] for i in i_cal])
                score_ms = 1000 * (time.perf_counter() - t1) / len(i_cal)
                S_te = score([U_alp[i] for i in i_te], "benign test")
                S_lms = score(U_lms, "benign shift")
                S_att = score(U_att, "attacks")

                # per-item scores
                rows = []
                for part, idx, S, items in (("cal", i_cal, S_cal, alpaca), ("test", i_te, S_te, alpaca)):
                    rows += [dict(item_id=items[i]["id"], part=part, family="benign", label=0, **s) for i, s in zip(idx, S)]
                rows += [dict(item_id=r["id"], part="shift", family="benign_shift", label=0, **s) for r, s in zip(lmsys, S_lms)]
                rows += [dict(item_id=r["id"], part="attack", family=r["family"], label=1, **s) for r, s in zip(attacks, S_att)]
                df = pd.DataFrame(rows)
                df.to_csv(out / "scores" / f"{enc.name}__seed{seed}.csv.gz", index=False)

                det, dec, cal_rows = [], [], []
                test = df[df.part == "test"]
                for fam in families + ["ALL"]:
                    att = df[(df.part == "attack") & ((df.family == fam) | (fam == "ALL"))]
                    if len(att) == 0:
                        continue
                    y = np.r_[np.ones(len(att)), np.zeros(len(test))].astype(int)
                    for m in measures:
                        s = np.r_[att[m].values, test[m].values]
                        det.append(dict(encoder=enc.name, seed=seed, family=fam, measure=m, **detection_metrics(y, s)))
                        c = ConformalCalibrator(cfg["alpha"]).fit(df[df.part == "cal"][m].values)
                        dec.append(dict(encoder=enc.name, seed=seed, family=fam, measure=m, alpha=cfg["alpha"],
                                        tau=c.tau, tpr=float(np.mean(att[m].values > c.tau)),
                                        fpr=float(np.mean(test[m].values > c.tau)),
                                        fpr_shift=float(np.mean(df[df.part == "shift"][m].values > c.tau))))
                cs = cfg["calibration_study"]
                for m in measures:
                    pool = df[df.part == "cal"][m].values
                    for shift, tgt in ((False, test[m].values), (True, df[df.part == "shift"][m].values)):
                        for r in coverage_study(pool, tgt, cs["alphas"], cs["n_cals"], cs["n_repeats"], seed):
                            cal_rows.append(dict(encoder=enc.name, seed=seed, measure=m, shift=shift, **r))
                append_rows(out / "detection.csv", det)
                append_rows(out / "decisions.csv", dec)
                append_rows(out / "calibration.csv", cal_rows)
                append_rows(out.parent / ("e5_overhead.csv" if not smoke else "e5_overhead_smoke.csv"),
                            [dict(device="pipeline", dim=enc.dim, measure=f"encode:{enc.name}", n_units=len(alpaca),
                                  median_ms=round(enc_ms, 3), p95_ms=""),
                             dict(device="pipeline", dim=enc.dim, measure=f"score_all8:{enc.name}", n_units=len(i_cal),
                                  median_ms=round(score_ms, 3), p95_ms="")])
                rec.update(n_benign=len(alpaca), n_shift=len(lmsys), n_attack=len(attacks), families=families)
                print(f"[E1] {enc.name} seed={seed} done: {len(det)} detection rows", flush=True)
            if not smoke:
                mark_done(marker(ec, seed), dict(encoder=enc.name))
            P.step(note=f"{enc.name} seed {seed}")
        free_gpu(enc)   # 8 GB GPUs hold one model at a time

    if not smoke:
        run_baselines(cfg, alpaca, lmsys, attacks, out)


def run_baselines(cfg, alpaca, lmsys, attacks, out):
    """LLM-based baselines. Same splits as WTS (benign test/cal indices from the same seed permutation)."""
    from wts import baselines as B
    from wts.llm import build_llm
    bc = cfg.get("baselines", {})
    items = alpaca + lmsys + attacks
    texts = [r["text"] for r in items]
    keys = [r["id"] for r in items]
    cols = {}
    ollama_unload_all()
    jdir = out / "scores" / "baseline_journal"     # one journal per baseline: a restart scores only missing items

    def jpath(name):
        return jdir / f"{name}_{tag_of(bc[name], 1)}.jsonl"
    if "perplexity" in bc:
        p = B.PerplexityScorer(bc["perplexity"]["model"])
        v = resumable_map("E1 baseline perplexity", keys, texts,
                          lambda b: list(zip(*[np.asarray(x).tolist() for x in p.scores(b)])), jpath("perplexity"))
        cols["ppl_mean"], cols["ppl_window"] = [x[0] for x in v], [x[1] for x in v]
        free_gpu(p)
    if "llama_guard" in bc:
        g = B.LlamaGuardScorer(bc["llama_guard"]["model"])
        t = time.perf_counter()
        n_before = len(texts) - sum(1 for _ in open(jpath("llama_guard"), encoding="utf-8")) \
            if jpath("llama_guard").exists() else len(texts)
        cols["llama_guard"] = resumable_map("E1 baseline Llama Guard", keys, texts,
                                            lambda b: list(g.scores(b)), jpath("llama_guard"))
        free_gpu(g)
        if n_before == len(texts):   # timing only from an uninterrupted pass over every item
            ms = 1000 * (time.perf_counter() - t) / len(texts)
            append_rows(out.parent / "e5_overhead.csv", [dict(device="cuda", dim="", measure="judge:llama_guard_3_8b",
                                                              n_units=len(texts), median_ms=round(ms, 3), p95_ms="")])
    base = pd.DataFrame(dict(item_id=[r["id"] for r in items], label=[r["label"] for r in items],
                             family=[r["family"] for r in items], **cols))
    if "pishield_probe" in bc:
        pc = bc["pishield_probe"]
        # training pool disjoint from every test item: deepset train split positives + fresh Alpaca rows
        tr_pos = [r for r in D.deepset_injections() if "-train-" in r["id"]]
        used = {r["text"] for r in alpaca}
        fresh = [r for r in D.alpaca_prompts(len(alpaca) + 3 * pc["train_n"], 99) if r["text"] not in used][: pc["train_n"]]
        pool = tr_pos[: pc["train_n"]] + fresh
        random.Random(0).shuffle(pool)
        k = int(0.8 * len(pool))
        probe = B.LastTokenProbe(pc["model"]).fit([r["text"] for r in pool[:k]], [r["label"] for r in pool[:k]],
                                                   [r["text"] for r in pool[k:]], [r["label"] for r in pool[k:]])
        test_ids = {r["id"] for r in pool}
        # a restart re-fits the probe on the same data; scores journaled by an earlier fit are kept only if the
        # re-fit selects the same layer (otherwise the journal is discarded and every item is re-scored)
        meta = jpath("pishield_probe").with_suffix(".layer")
        if meta.exists() and meta.read_text().strip() != str(probe.layer):
            jpath("pishield_probe").unlink(missing_ok=True)
        jdir.mkdir(parents=True, exist_ok=True)
        meta.write_text(str(probe.layer))
        base["pishield"] = resumable_map("E1 baseline PIShield probe", keys, texts, lambda b: list(probe.scores(b)),
                                         jpath("pishield_probe"))
        free_gpu(probe)
        base["pishield_train_overlap"] = base.item_id.isin(test_ids)   # deepset-train rows: excluded in tables
    if "known_answer" in bc:   # Ollama / vLLM: run last, after every in-process model has been freed
        ka = build_llm(bc["known_answer"]["llm"])
        base["known_answer"] = resumable_map("E1 baseline known-answer", keys, texts,
                                             lambda b: list(B.known_answer_scores(b, ka)), jpath("known_answer"), chunk=8)
    base.to_csv(out / "scores" / "baselines.csv.gz", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/e1.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(yaml.safe_load(open(args.config)), smoke=args.smoke)
