"""results/*.csv -> paper/figures/*.pdf (Figures 2-6). Figure 1 (architecture) is drawn by hand in
paper/figures/fig1_architecture.tex. Each figure is skipped when its inputs do not exist.

  python scripts/make_figures.py --results results --out paper/figures
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from make_tables import refuse_smoke_into_paper  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
MEAS_LABEL = {"w2_gauss": "Gaussian W$_2$ (ours)", "w2_sliced": "Sliced W$_2$ (ours)", "mahalanobis": "Mahalanobis",
              "cosine": "Cosine", "euclidean": "Euclidean", "mmd_rbf": "MMD (RBF)"}
SKIPPED = []


def fig2_terms(res, out):
    files = sorted(glob.glob(str(res / "e1/scores/*__seed0.csv.gz")))
    if not files:
        return SKIPPED.append("Figure 2")
    d = pd.read_csv(files[0])
    d = d[d.part.isin(["test", "attack"])]
    d["shape_share"] = d.shape_sq / (d.loc_sq + d.shape_sq)
    fams = sorted(d.family.unique())
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.violinplot([d[d.family == f].shape_share.values for f in fams], showmedians=True)
    ax.set_xticks(range(1, len(fams) + 1), [f.replace("_", "\n") for f in fams], fontsize=7)
    ax.set_ylabel("shape term / W$_2^2$")
    ax.set_title(f"Share of the W$_2$ signal in the covariance (shape) term - {Path(files[0]).name.split('__')[0]}",
                 fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(out / "fig2_location_vs_shape.pdf"); plt.close(fig)


def fig3_roc(res, out):
    files = sorted(glob.glob(str(res / "e1/scores/*__seed0.csv.gz")))
    if not files:
        return SKIPPED.append("Figure 3")
    fig, axes = plt.subplots(1, len(files), figsize=(3.2 * len(files), 3), squeeze=False)
    for ax, f in zip(axes[0], files):
        d = pd.read_csv(f)
        d = d[d.part.isin(["test", "attack"])]
        for m, lab in MEAS_LABEL.items():
            fpr, tpr, _ = roc_curve(d.label, d[m])
            ax.plot(fpr, tpr, lw=2 if m.startswith("w2") else 1, label=lab)
        ax.plot([0, 1], [0, 1], ":", c="grey", lw=0.8)
        ax.set_xscale("log"); ax.set_xlim(1e-3, 1)
        ax.set_xlabel("FPR (log)"); ax.set_ylabel("TPR")
        ax.set_title(Path(f).name.split("__")[0], fontsize=9, loc="left")
    axes[0][-1].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig3_roc_e1.pdf"); plt.close(fig)


def fig4_calibration(res, out):
    p = res / "e1/calibration.csv"
    if not p.exists():
        return SKIPPED.append("Figure 4")
    c = pd.read_csv(p)
    c = c[c.measure == ("w2_gauss_adj" if "w2_gauss_adj" in set(c.measure) else "w2_gauss")]
    enc = sorted(c.encoder.unique())[0]
    c = c[c.encoder == enc]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6), sharey=True)
    for ax, shift in zip(axes, (False, True)):
        g = c[c["shift"] == shift].groupby(["alpha", "n_cal"]).agg(m=("fpr_mean", "mean"), lo=("fpr_q05", "mean"),
                                                                    hi=("fpr_q95", "mean")).reset_index()
        for a, gg in g.groupby("alpha"):
            ax.errorbar(gg.n_cal, gg.m, yerr=[np.clip(gg.m - gg.lo, 0, None), np.clip(gg.hi - gg.m, 0, None)], marker="o", ms=3, capsize=2,
                        label=f"$\\alpha$={a}")
            ax.axhline(a, ls="--", lw=0.7, c=ax.lines[-1].get_color())
        ax.set_xscale("log"); ax.set_xlabel("calibration size $n$")
        ax.set_title("benign test from calibration source" if not shift else "benign test under shift (LMSYS)",
                     fontsize=9, loc="left")
    axes[0].set_ylabel("empirical FPR"); axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig4_calibration.pdf"); plt.close(fig)


def fig5_poison_count(res, out):
    p = res / "e2/items.csv"
    if not p.exists():
        return SKIPPED.append("Figure 5")
    it = pd.read_csv(p)
    it = it[it.attack.str.match(r"poisonedrag_(n\d|blackbox)")]
    it["n"] = it.attack.map(lambda a: 5 if a == "poisonedrag_blackbox" else int(a[-1]))
    g = it.groupby(["dataset", "defense", "n"]).incorrect_hit.mean().reset_index()
    dss = sorted(g.dataset.unique())
    fig, axes = plt.subplots(1, len(dss), figsize=(2.3 * len(dss), 2.5), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], dss):
        for dfn, gg in g[g.dataset == ds].groupby("defense"):
            ax.plot(gg.n, gg.incorrect_hit, marker="o", ms=3, lw=2 if dfn == "wts" else 1, label=dfn)
        ax.set_title(ds, fontsize=9, loc="left"); ax.set_xlabel("poisoned passages")
    axes[0][0].set_ylabel("ASR"); axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig5_asr_vs_poison_count.pdf"); plt.close(fig)


def fig6_evasion(res, out):
    p, q = res / "e4/dilution.csv", res / "e4/evasion_cost.csv"
    if not p.exists():
        return SKIPPED.append("Figure 6")
    d = pd.read_csv(p)
    base = d[d.factor == 0][["seed", "item", "flagged", "score"]].rename(columns={"flagged": "f0", "score": "s0"})
    m = d.merge(base, on=["seed", "item"])
    m = m[m.f0]
    g = m.groupby("factor").agg(ev=("flagged", lambda x: 1 - x.mean()), w=("added_words", "median")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    axes[0].plot(g.w, g.ev, marker="o", ms=3)
    axes[0].set_xlabel("added words (median)"); axes[0].set_ylabel("share evading WTS")
    axes[0].set_title("Evasion vs dilution cost", fontsize=9, loc="left")
    if q.exists():
        c = pd.read_csv(q)
        c = c[c.flagged0]
        c["cost"] = c.evaded_at_factor * c.words
        ok = c.dropna(subset=["cost"])
        axes[1].scatter(ok.gap, ok.cost, s=4, alpha=0.5, label="evaded (words added)")
        axes[1].set_xlabel("score gap $s(x)-\\tau$"); axes[1].set_ylabel("words added to evade")
        axes[1].set_title("Proposition 2: cost grows with the gap", fontsize=9, loc="left")
        if len(ok) > 2:
            k = np.polyfit(ok.gap, ok.cost, 1)
            xs = np.linspace(ok.gap.min(), ok.gap.max(), 50)
            axes[1].plot(xs, np.polyval(k, xs), c="k", lw=0.8, label="linear fit")
        axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig6_evasion_cost.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "paper/figures"))
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    refuse_smoke_into_paper(res, out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig2_terms, fig3_roc, fig4_calibration, fig5_poison_count, fig6_evasion):
        f(res, out)
    print("written:", sorted(p.name for p in out.glob("*.pdf")), "| skipped (no inputs):", SKIPPED)


if __name__ == "__main__":
    main()
