"""results/*.csv -> paper/tables/*.tex and results/verdicts.md (genuineness rule 1: numbers are never typed).

Each table is skipped (with a note) when its inputs do not exist yet. Confidence intervals: cluster bootstrap
over items (all seeds stacked, items resampled with all their rows, stratified by label), n_boot = 10,000
by default. Hypothesis verdicts are computed against the pass criteria frozen in hypotheses.md.

  python scripts/make_tables.py --results results --out paper/tables
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from wts.measures import LOCATION_ONLY  # noqa: E402
from wts.stats import holm, relative_reduction, wilcoxon_paired  # noqa: E402

NOTES: list[str] = []
VERDICTS: list[str] = []


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def ci(est, lo, hi, d=3):
    return f"{fmt(est, d)} [{fmt(lo, d)}, {fmt(hi, d)}]"


def write_tex(path: Path, caption: str, label: str, df: pd.DataFrame):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = df.to_latex(index=False, escape=True, column_format="l" * len(df.columns))
    path.write_text("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + caption + "}\n\\label{" + label + "}\n"
                    + body + "\\end{table}\n")


# ------------------------------------------------------------------------------ cluster bootstrap AUROC
def _stack_scores(score_files):
    dfs = []
    for f in score_files:
        d = pd.read_csv(f)
        d["seed"] = int(Path(f).name.split("seed")[1].split(".")[0])
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def cluster_boot_auroc(df, cols, n_boot, seed=0):
    """AUROC for each column in cols (and paired differences vs cols[0]) with item-cluster bootstrap."""
    rng = np.random.default_rng(seed)
    ids = df.item_id.values
    pos_ids = np.unique(ids[df.label.values == 1])
    neg_ids = np.unique(ids[df.label.values == 0])
    rows_of = df.groupby("item_id").indices
    y = df.label.values
    S = {c: df[c].values for c in cols}
    point = {c: roc_auc_score(y, S[c]) for c in cols}
    boots = {c: np.empty(n_boot) for c in cols}
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(pos_ids, len(pos_ids)), rng.choice(neg_ids, len(neg_ids))])
        idx = np.concatenate([rows_of[i] for i in pick])
        for c in cols:
            boots[c][b] = roc_auc_score(y[idx], S[c][idx])
    return point, boots


# ------------------------------------------------------------------------------ Table 2 + H1
def table_e1(res: Path, out: Path, n_boot: int):
    files = sorted(glob.glob(str(res / "e1/scores/*__seed*.csv.gz")))
    if not files:
        NOTES.append("Table 2 / H1 skipped: no results/e1/scores files.")
        return
    encoders = sorted({Path(f).name.split("__")[0] for f in files})
    det = pd.read_csv(res / "e1/detection.csv")
    rows = []
    for variant, label in (("", "raw scores (original pre-registration)"), ("_adj", "length-adjusted (addendum A1)")):
        measures = [m + variant for m in ["w2_gauss", "w2_sliced", "mmd_rbf", *LOCATION_ONLY]]
        loc_only = [m + variant for m in LOCATION_ONLY]
        h1 = []
        for enc in encoders:
            df = _stack_scores([f for f in files if Path(f).name.startswith(enc + "__")])
            df = df[df.part.isin(["test", "attack"])]
            if not set(measures) <= set(df.columns):
                NOTES.append(f"H1 {label}: columns missing for {enc}")
                continue
            point, boots = cluster_boot_auroc(df, measures, n_boot)
            d_all = det[(det.encoder == enc) & (det.family == "ALL")].groupby("measure").mean(numeric_only=True)
            for m in measures:
                lo, hi = np.quantile(boots[m], [0.025, 0.975])
                rows.append({"Encoder": enc, "Score": m, "AUROC [95% CI]": ci(point[m], lo, hi),
                             "AUPRC": fmt(d_all.loc[m, "auprc"]), "TPR@1%FPR": fmt(d_all.loc[m, "tpr_at_1fpr"]),
                             "FPR@95%TPR": fmt(d_all.loc[m, "fpr_at_95tpr"])})
            best_loc = max(loc_only, key=lambda m: point[m])
            diff = boots["w2_gauss" + variant] - boots[best_loc]
            lo, hi = np.quantile(diff, [0.025, 0.975])
            obs = point["w2_gauss" + variant] - point[best_loc]
            h1.append(dict(encoder=enc, best_location=best_loc, gain=obs, lo=lo, hi=hi,
                           passes=(obs >= 0.03) and (lo > 0)))
        if h1:
            n_pass = sum(h["passes"] for h in h1)
            VERDICTS.append(f"H1, {label}: {'SUPPORTED' if n_pass >= 2 else 'NOT SUPPORTED'} "
                            f"(AUROC gain >= 0.03 with CI excluding 0 on {n_pass}/{len(h1)} encoders; criterion: >= 2 of 3)")
            for h in h1:
                VERDICTS.append(f"  - {h['encoder']}: {'w2_gauss' + variant} - {h['best_location']} = "
                                f"{h['gain']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}]")
    write_tex(out / "table2_e1_detection.tex",
              "Prompt-channel detection (E1), all attack families pooled; 5 seeds stacked, 95\\% item-cluster "
              "bootstrap CI. Suffix \\texttt{\\_adj}: length-adjusted (addendum A1).",
              "tab:e1", pd.DataFrame(rows))


# ------------------------------------------------------------------------------ Table 7 (ablation) + RQ4 table + H4
def table_ablation_and_calibration(res: Path, out: Path):
    p = res / "e1/detection.csv"
    if not p.exists():
        NOTES.append("Table 7 / H4 skipped: no results/e1/detection.csv.")
        return
    det = pd.read_csv(p)
    ab = det[det.measure.isin(["w2_gauss", "w2_location", "w2_shape", "w2_sliced", "mahalanobis"])]
    piv = ab.groupby(["encoder", "family", "measure"]).auroc.mean().unstack("measure").reset_index()
    for c in piv.columns[2:]:
        piv[c] = piv[c].map(fmt)
    write_tex(out / "table7_ablation.tex", "Ablation of the score by attack family (E1): AUROC, mean over seeds.",
              "tab:ablation", piv)
    cp = res / "e1/calibration.csv"
    if not cp.exists():
        return
    cal = pd.read_csv(cp)
    c = cal[cal.measure == ("w2_gauss_adj" if "w2_gauss_adj" in set(cal.measure) else "w2_gauss")].groupby(["encoder", "shift", "alpha", "n_cal"]).agg(
        fpr=("fpr_mean", "mean"), inf=("frac_infinite_tau", "mean")).reset_index()
    t = c.pivot_table(index=["encoder", "alpha", "n_cal"], columns="shift", values="fpr").reset_index()
    t.columns = ["Encoder", "alpha", "n_cal", "FPR in-dist", "FPR shift"]
    viol = t[(t["FPR in-dist"] > t["alpha"] + 0.01) & (t["n_cal"] >= np.ceil(1 / t["alpha"]))]
    gap = (t["FPR shift"] - t["alpha"]).max()
    for col in ("FPR in-dist", "FPR shift"):
        t[col] = t[col].map(fmt)
    write_tex(out / "table_rq4_calibration.tex", "Empirical false-positive rate of the conformal threshold (W$_2$).",
              "tab:calib", t)
    VERDICTS.append(f"H4: {'SUPPORTED' if len(viol) == 0 else 'NOT SUPPORTED'} "
                    f"({len(viol)} (encoder, alpha, n_cal) cells exceed alpha + 0.01 in distribution); "
                    f"largest shift excess over alpha = {gap:+.3f}")


# ------------------------------------------------------------------------------ Table 3 + H2
def table_e2(res: Path, out: Path):
    p = res / "e2/items.csv"
    if not p.exists():
        NOTES.append("Table 3 / H2 skipped: no results/e2/items.csv.")
        return
    it = pd.read_csv(p)
    clean = it[it.attack == "clean"].groupby(["dataset", "retriever", "llm", "defense"]).correct_hit.mean()
    att = it[it.attack == "poisonedrag_blackbox"].groupby(["dataset", "retriever", "llm", "defense"]).incorrect_hit.mean()
    rows, ok = [], []
    for (ds, ret, llm), g in att.groupby(level=[0, 1, 2]):
        none = g.xs("none", level=3).iloc[0] if "none" in g.index.get_level_values(3) else np.nan
        acc_none = clean.get((ds, ret, llm, "none"), np.nan)
        for dfn in g.index.get_level_values(3):
            a = g.xs(dfn, level=3).iloc[0]
            acc = clean.get((ds, ret, llm, dfn), np.nan)
            rows.append({"Dataset": ds, "LLM": llm.split("/")[-1], "Defense": dfn, "ASR": fmt(a),
                         "Rel. reduction": fmt(relative_reduction(none, a)), "Clean acc.": fmt(acc),
                         "Acc. loss": fmt(acc_none - acc)})
            if dfn == "wts":
                ok.append(relative_reduction(none, a) >= 0.5 and (acc_none - acc) <= 0.02)
    write_tex(out / "table3_e2_rag.tex", "RAG poisoning (E2): attack success rate and clean accuracy.", "tab:e2",
              pd.DataFrame(rows))
    VERDICTS.append(f"H2: {'SUPPORTED' if ok and all(ok) else 'NOT SUPPORTED'} "
                    f"(criterion met in {sum(ok)}/{len(ok)} dataset x retriever x LLM cells)")


# ------------------------------------------------------------------------------ Table 4 + H3
def table_e3(res: Path, out: Path):
    p = res / "e3/agentdojo_items.csv"
    rows, ok = [], []
    if p.exists():
        it = pd.read_csv(p)
        ben = it[it.attack == "none"].groupby(["llm", "defense"]).utility.mean()
        att = it[it.attack == "important_instructions"].groupby(["llm", "defense"]).agg(
            uua=("utility", "mean"), asr=("security", "mean"))
        for (llm, dfn), r in att.iterrows():
            none_asr = att.loc[(llm, "none"), "asr"]
            red = relative_reduction(none_asr, r.asr)
            loss = ben[(llm, "none")] - ben[(llm, dfn)]
            rows.append({"Benchmark": "AgentDojo", "LLM": llm.split("/")[-1], "Defense": dfn,
                         "Utility": fmt(ben[(llm, dfn)]), "Utility under attack": fmt(r.uua),
                         "Targeted ASR": fmt(r.asr), "Rel. reduction": fmt(red)})
            if dfn == "wts_full":
                ok.append(red >= 0.4 and loss <= 0.05)
    q = res / "e3/injecagent_items.csv"
    if q.exists():
        ia = pd.read_csv(q)
        for (llm, setting, dfn), g in ia.groupby(["llm", "setting", "defense"]):
            v = g[g["eval"].isin(["succ", "unsucc"])]
            rows.append({"Benchmark": f"InjecAgent-{setting}", "LLM": llm.split("/")[-1], "Defense": dfn,
                         "Utility": "--", "Utility under attack": "--",
                         "Targeted ASR": fmt((v["eval"] == "succ").mean() if len(v) else np.nan),
                         "Rel. reduction": "--"})
    if not rows:
        NOTES.append("Table 4 / H3 skipped: no E3 agent results.")
        return
    write_tex(out / "table4_e3_agents.tex", "Agents (E3): utility and targeted attack success.", "tab:e3",
              pd.DataFrame(rows))
    if ok:
        VERDICTS.append(f"H3 (AgentDojo part): {'SUPPORTED' if all(ok) else 'NOT SUPPORTED'} "
                        f"({sum(ok)}/{len(ok)} LLMs meet >=40% reduction with <=5 pt utility loss)")


# ------------------------------------------------------------------------------ Table 5 + H5, Table 6 + H6
def table_e4(res: Path, out: Path):
    p = res / "e4/dilution.csv"
    if not p.exists():
        NOTES.append("Table 5 skipped: no results/e4/dilution.csv.")
        return
    d = pd.read_csv(p)
    base = d[d.factor == 0][["seed", "item", "flagged"]].rename(columns={"flagged": "f0"})
    m = d.merge(base, on=["seed", "item"])
    m = m[m.f0]
    t = m.groupby("factor").agg(still_flagged=("flagged", "mean"), added_words=("added_words", "median")).reset_index()
    t["still_flagged"] = t.still_flagged.map(fmt)
    write_tex(out / "table5_e4_dilution.tex",
              "Dilution attack (E4): share of initially flagged attacks still flagged, by dilution factor.",
              "tab:e4", t)
    e2 = res / "e2/items.csv"
    if e2.exists():
        it = pd.read_csv(e2)
        w = it[it.defense.isin(["none", "wts"])].groupby(["attack", "defense"]).incorrect_hit.mean().unstack()
        if {"poisonedrag_blackbox"} <= set(w.index):
            none, dfd = w.loc["poisonedrag_blackbox", "none"], w.loc["poisonedrag_blackbox", "wts"]
            gap = none - dfd
            for a in [x for x in w.index if "dilute" in x or "paraphrase" in x]:
                rec = (w.loc[a, "wts"] - dfd) / gap if gap else np.nan
                VERDICTS.append(f"H5 (RAG, {a}): recovered ASR = {fmt(rec)} "
                                f"({'>= 0.5: partial evasion as expected' if rec >= 0.5 else '< 0.5'})")


def table_e5(res: Path, out: Path):
    p = res / "e5_overhead.csv"
    if not p.exists():
        NOTES.append("Table 6 / H6 skipped: no results/e5_overhead.csv.")
        return
    t = pd.read_csv(p)
    write_tex(out / "table6_overhead.tex", "Per-unit cost (ms).", "tab:e5", t)
    g = t[(t.measure == "w2_gauss") & (t.device.astype(str).str.startswith("cuda")) & (t.dim == 768)]
    j = t[t.measure.astype(str).str.startswith("judge:")]
    if len(g) and len(j):
        fast = g.median_ms.min() < 20
        ratio = float(j.median_ms.iloc[0]) / float(g.median_ms.min())
        VERDICTS.append(f"H6: {'SUPPORTED' if fast and ratio >= 10 else 'NOT SUPPORTED'} "
                        f"(GPU W2 median {g.median_ms.min():.2f} ms at d=768; judge/W2 = {ratio:.1f}x)")
    else:
        NOTES.append("H6 not evaluated: needs a cuda W2 row and a judge row in e5_overhead.csv.")


def refuse_smoke_into_paper(res: Path, out: Path):
    """Smoke results or test-only encoders must never reach paper/ (genuineness rule 1)."""
    into_paper = (REPO / "paper") in out.resolve().parents or out.resolve() == REPO / "paper"
    tainted = "_smoke" in res.parts or any("test-only" in f for f in glob.glob(str(res / "**/*"), recursive=True))
    if into_paper and tainted:
        raise SystemExit("Refusing to write tables into paper/ from smoke or test-only-encoder results.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "paper/tables"))
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    refuse_smoke_into_paper(res, out)
    table_e1(res, out, a.n_boot)
    table_ablation_and_calibration(res, out)
    table_e2(res, out)
    table_e3(res, out)
    table_e4(res, out)
    table_e5(res, out)
    txt = "# Hypothesis verdicts (generated by scripts/make_tables.py)\n\n" + "\n".join(VERDICTS or ["(none yet)"])
    txt += "\n\n## Skipped\n\n" + "\n".join(f"- {n}" for n in NOTES)
    (res / "verdicts.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
