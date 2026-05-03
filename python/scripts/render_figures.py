"""Generate the headline visualizations as standalone PNG files.

Produces five plots that summarize the project's findings without needing
Jupyter or a notebook reader:

1. mnl_attenuation.png      - True vs MNL coefficients (the bias notebook 01 shows)
2. mxl_recovery.png         - True vs MNL vs MXL with 95% CIs (notebook 02)
3. elasticity_heatmap.png   - MNL vs MXL elasticity matrices (notebook 03)
4. wtp_distributions.png    - WTP histograms with median + IQR (notebook 04)
5. pricing_projection.png   - The 5% Premium hike: MNL vs MXL share shifts

Run from python/:
    python scripts/render_figures.py

Outputs go to ../docs/figures/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from mixedlogit.dgp import default_config, simulate_choices
from mixedlogit.elasticity import (
    mnl_aggregate_elasticities,
    mxl_aggregate_elasticities,
)
from mixedlogit.halton import standard_normal_draws
from mixedlogit.mnl import fit_mnl
from mixedlogit.mxl import _build_param_layout, _draw_betas, fit_mxl
from mixedlogit.wtp import compute_wtp_samples


# Matplotlib style: clean, presentation-ready
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "x",
    "grid.alpha": 0.25,
    "font.size": 10,
})

COLOR_TRUTH = "#4C78A8"
COLOR_MNL = "#F58518"
COLOR_MXL = "#54A24B"


def _outdir() -> Path:
    here = Path(__file__).resolve().parent
    out = here.parents[1] / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fit_models(df, cfg):
    print("  fitting MNL...")
    mnl = fit_mnl(df, attr_names=["price", "quality", "brand_known"])
    print("  fitting MXL (~30s)...")
    mxl = fit_mxl(df, cfg.attributes, n_draws=200, halton_seed=0)
    return mnl, mxl


# --- Plot 1: MNL attenuation -------------------------------------------------


def plot_mnl_attenuation(mnl, out: Path) -> Path:
    truth = {"price": -1.2, "quality": 0.8, "brand_known": 0.6}
    names = mnl.coef_names
    est = dict(zip(names, mnl.coefficients, strict=True))
    se = dict(zip(names, mnl.std_errors, strict=True))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(names))
    h = 0.38
    ax.barh(y - h / 2, [truth[n] for n in names], h,
            label="True population mean", color=COLOR_TRUTH, alpha=0.95)
    ax.barh(y + h / 2, [est[n] for n in names], h,
            label="MNL estimate (95% CI)", color=COLOR_MNL, alpha=0.9,
            xerr=[1.96 * se[n] for n in names], capsize=4,
            error_kw={"ecolor": "#333", "lw": 1})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("coefficient value")
    ax.set_title("MNL attenuation bias on heterogeneous data\n"
                 "Random-coefficient attributes (price, quality) are pulled toward zero")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = out / "mnl_attenuation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# --- Plot 2: MXL recovery ----------------------------------------------------


def plot_mxl_recovery(mnl, mxl, out: Path) -> Path:
    common = ["price", "quality", "brand_known"]
    truth = {"price": -1.2, "quality": 0.8, "brand_known": 0.6}
    mnl_est = dict(zip(mnl.coef_names, mnl.coefficients, strict=True))
    mnl_se = dict(zip(mnl.coef_names, mnl.std_errors, strict=True))
    est = dict(zip(mxl.coef_names, mxl.coefficients, strict=True))
    se = dict(zip(mxl.coef_names, mxl.std_errors, strict=True))
    mxl_means = {
        "price": (est["price [mean]"], se["price [mean]"]),
        "quality": (est["quality [mean]"], se["quality [mean]"]),
        "brand_known": (est["brand_known"], se["brand_known"]),
    }

    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(common))
    h = 0.28
    ax.barh(y - h, [truth[a] for a in common], h,
            label="True population mean", color=COLOR_TRUTH, alpha=0.95)
    ax.barh(y, [mnl_est[a] for a in common], h,
            label="MNL estimate", color=COLOR_MNL, alpha=0.9,
            xerr=[1.96 * mnl_se[a] for a in common], capsize=4,
            error_kw={"ecolor": "#333", "lw": 1})
    ax.barh(y + h, [mxl_means[a][0] for a in common], h,
            label="MXL estimate", color=COLOR_MXL, alpha=0.9,
            xerr=[1.96 * mxl_means[a][1] for a in common], capsize=4,
            error_kw={"ecolor": "#333", "lw": 1})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(common)
    ax.set_xlabel("coefficient value")
    ax.set_title("MXL closes the bias MNL leaves open\n"
                 "(error bars = 95% CI)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = out / "mxl_recovery.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# --- Plot 3: elasticity heatmaps ---------------------------------------------


def plot_elasticity_heatmap(mnl, mxl, df, out: Path) -> Path:
    design = np.array([
        [2.0, 1.0, 1.0],   # Premium
        [1.0, 0.0, 1.0],   # Mid
        [0.5, -1.0, 0.0],  # Budget
    ])
    labels = ["Premium", "Mid", "Budget"]
    mnl_E = mnl_aggregate_elasticities(mnl, df, price_attr="price",
                                       design=design, alt_labels=labels).matrix
    mxl_E = mxl_aggregate_elasticities(mxl, df, price_attr="price",
                                       design=design, alt_labels=labels,
                                       n_draws=2000, halton_seed=0).matrix

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    norm = TwoSlopeNorm(vmin=-1.6, vcenter=0.0, vmax=0.8)
    for ax, E, title in zip(
        axes, [mnl_E, mxl_E],
        ["MNL — IIA imposed (column off-diagonals identical)",
         "MXL — IIA-free (Mid is closer substitute to Premium than Budget)"],
        strict=True,
    ):
        im = ax.imshow(E, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("price source (j)")
        ax.set_ylabel("share affected (i)")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{E[i, j]:+.3f}",
                        ha="center", va="center", fontsize=10,
                        color="black", fontweight="bold")
    fig.colorbar(im, ax=axes, label="elasticity", shrink=0.85, pad=0.03)
    fig.suptitle("Aggregate elasticity matrices on the Premium / Mid / Budget design",
                 y=1.0, fontsize=12)

    path = out / "elasticity_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# --- Plot 4: WTP distributions -----------------------------------------------


def plot_wtp_distributions(mxl, out: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wtps = compute_wtp_samples(mxl, n_draws=20_000, halton_seed=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    truth_map = {"quality": 0.8 / 1.2, "brand_known": 0.6 / 1.2}
    for ax, (name, w) in zip(axes, wtps.items(), strict=True):
        lo, hi = w.quantile([0.01, 0.99])
        s = w.samples[(w.samples >= lo) & (w.samples <= hi)]
        ax.hist(s, bins=60, alpha=0.7, color=COLOR_MXL, edgecolor="white")
        ax.axvline(w.median, color=COLOR_MNL, lw=2,
                   label=f"Median: {w.median:.3f}")
        p25, p75 = w.quantile([0.25, 0.75])
        ax.axvspan(p25, p75, alpha=0.15, color=COLOR_MNL,
                   label=f"IQR: [{p25:.2f}, {p75:.2f}]")
        true_val = truth_map[name]
        ax.axvline(true_val, color=COLOR_TRUTH, lw=2, ls="--",
                   label=f"True median: {true_val:.3f}")
        ax.set_title(f"WTP for {name}")
        ax.set_xlabel("WTP (in price units)")
        ax.set_ylabel("frequency")
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Recovered willingness-to-pay distributions",
                 y=1.02, fontsize=12)
    plt.tight_layout()

    path = out / "wtp_distributions.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# --- Plot 5: pricing projection (the decision-making artifact) --------------


def plot_pricing_projection(mnl, mxl, df, out: Path) -> Path:
    design = np.array([
        [2.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, -1.0, 0.0],
    ])
    labels = ["Premium", "Mid", "Budget"]
    mnl_E = mnl_aggregate_elasticities(mnl, df, design=design,
                                       alt_labels=labels).matrix
    mxl_E = mxl_aggregate_elasticities(mxl, df, design=design,
                                       alt_labels=labels,
                                       n_draws=2000, halton_seed=0).matrix

    # Aggregate shares under MXL at the design
    attr_specs = mxl.attr_specs
    n_attr = len(attr_specs)
    _, layout = _build_param_layout(attr_specs)
    n_random = sum(1 for _, kind, _ in layout if kind != "fixed")
    z = standard_normal_draws(2000, n_random, seed=0)[None, :, :]
    betas = _draw_betas(mxl.coefficients, layout, 1, n_attr, z)[0]
    V = (design @ betas.T).T
    V -= V.max(axis=1, keepdims=True)
    P = np.exp(V) / np.exp(V).sum(axis=1, keepdims=True)
    shares = P.mean(axis=0)

    delta_p = 0.05  # 5% Premium price increase
    rows = []
    for i, alt in enumerate(labels):
        mnl_dshare = mnl_E[i, 0] * delta_p * shares[i] * 100  # in pp
        mxl_dshare = mxl_E[i, 0] * delta_p * shares[i] * 100
        rows.append({"alt": alt, "MNL": mnl_dshare, "MXL": mxl_dshare})
    df_proj = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, df_proj["MNL"], w, label="MNL projection",
           color=COLOR_MNL, alpha=0.9)
    ax.bar(x + w / 2, df_proj["MXL"], w, label="MXL projection",
           color=COLOR_MXL, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Δ market share (percentage points)")
    ax.set_title("Projected share impact of a 5% Premium price increase\n"
                 "MNL vs MXL — different stories for the spillover")
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", alpha=0.25)
    for i, (mnl_v, mxl_v) in enumerate(zip(df_proj["MNL"], df_proj["MXL"], strict=True)):
        ax.text(i - w / 2, mnl_v + (0.05 if mnl_v >= 0 else -0.15),
                f"{mnl_v:+.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, mxl_v + (0.05 if mxl_v >= 0 else -0.15),
                f"{mxl_v:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()

    path = out / "pricing_projection.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path.name}")
    return path


# --- Driver ------------------------------------------------------------------


def main() -> None:
    print("Generating headline figures...")
    cfg = default_config()
    df = simulate_choices(cfg)
    mnl, mxl = _fit_models(df, cfg)

    out = _outdir()
    print(f"Writing to {out}")

    plot_mnl_attenuation(mnl, out)
    plot_mxl_recovery(mnl, mxl, out)
    plot_elasticity_heatmap(mnl, mxl, df, out)
    plot_wtp_distributions(mxl, out)
    plot_pricing_projection(mnl, mxl, df, out)
    print("Done.")


if __name__ == "__main__":
    main()
