"""
06_figures.py — Publication-quality figures (300 DPI, colorblind-safe).

Generates:
    fig1: Validity leaderboard heatmap (benchmark x failure mode)
    fig2: Native vs MIRAGE-Full pass rates (grouped bar + bootstrap CIs)
    fig3: CDVA commutator distribution (violin plot per model, tau line)
    fig4: Failure mode stacked bars per benchmark
    fig5: Commutator magnitude by demographic axis
    fig6: MeasDefect distribution per model (CDF)

Outputs to: outputs/figures/
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.figsize": (8, 5),
})

PALETTE = sns.color_palette("cividis", 4)
MODEL_SHORT = {
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "qwen2.5-7b-instruct": "Qwen2.5-7B",
    "gemma-2-2b-it": "Gemma-2-2B",
    "phi-4-mini-instruct": "Phi-4-mini",
}
BENCH_LABELS = {"bbq": "BBQ", "crows_pairs": "CrowS-Pairs", "stereoset": "StereoSet"}


def fig1_leaderboard_heatmap():
    """Validity leaderboard heatmap (benchmark x failure mode)."""
    lb = pd.read_parquet(OUTPUT_DIR / "leaderboard_matrix.parquet")
    lb = lb.set_index("benchmark")
    lb.index = lb.index.map(lambda x: BENCH_LABELS.get(x, x))
    fm_cols = ["FM1", "FM2", "FM3", "FM4", "FM5"]

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(lb[fm_cols], annot=True, fmt=".3f", cmap="YlOrRd",
                vmin=0, vmax=0.5, ax=ax, linewidths=0.5)
    ax.set_title("Failure Mode Rates by Source Benchmark")
    ax.set_ylabel("")
    ax.set_xlabel("Failure Mode")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_leaderboard_heatmap.png")
    plt.close()
    print("  fig1_leaderboard_heatmap.png")


def fig2_validity_gap_bars():
    """Native vs MIRAGE-Full grouped bars with bootstrap CIs."""
    ci = pd.read_parquet(OUTPUT_DIR / "bootstrap_cis.parquet")
    ci["model_short"] = ci["model_name"].map(MODEL_SHORT)
    ci["bench_label"] = ci["benchmark"].map(BENCH_LABELS)

    fig, axes = plt.subplots(1, len(ci["benchmark"].unique()), figsize=(14, 5), sharey=True)
    if len(ci["benchmark"].unique()) == 1:
        axes = [axes]

    for idx, bench in enumerate(sorted(ci["benchmark"].unique())):
        ax = axes[idx]
        sub = ci[ci["benchmark"] == bench].sort_values("model_short")
        x = np.arange(len(sub))
        w = 0.35

        ax.bar(x - w/2, sub["native_point"], w, label="Native",
               color=PALETTE[0], alpha=0.85,
               yerr=[sub["native_point"] - sub["native_ci_lo"],
                     sub["native_ci_hi"] - sub["native_point"]],
               capsize=3)
        ax.bar(x + w/2, sub["mirage_full_point"], w, label="MIRAGE-Full",
               color=PALETTE[2], alpha=0.85,
               yerr=[sub["mirage_full_point"] - sub["mirage_full_ci_lo"],
                     sub["mirage_full_ci_hi"] - sub["mirage_full_point"]],
               capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_short"], rotation=30, ha="right", fontsize=9)
        ax.set_title(BENCH_LABELS.get(bench, bench))
        ax.set_ylim(0, 1.0)
        if idx == 0:
            ax.set_ylabel("Pass Rate")
            ax.legend(fontsize=9)

    plt.suptitle("Native vs MIRAGE-Full Pass Rates (95% Bootstrap CI)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_validity_gap_bars.png")
    plt.close()
    print("  fig2_validity_gap_bars.png")


def fig3_cdva_violin():
    """CDVA commutator distribution violin plot per model."""
    cdva = pd.read_parquet(OUTPUT_DIR / "cdva_commutators.parquet")
    cdva["model_short"] = cdva["model_name"].map(MODEL_SHORT)

    with open(OUTPUT_DIR / "cdva_summary.json") as f:
        summary = json.load(f)
    tau = summary["tau"]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.violinplot(data=cdva, x="model_short", y="commutator_mag",
                   palette="cividis", inner="quartile", ax=ax, cut=0)
    ax.axhline(tau, color="red", linestyle="--", linewidth=1.2, label=f"τ = {tau:.3f}")
    ax.set_xlabel("Model")
    ax.set_ylabel("|Commutator| = |Δlogit|")
    ax.set_title("CDVA Commutator Magnitude Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_cdva_violin.png")
    plt.close()
    print("  fig3_cdva_violin.png")


def fig4_failure_mode_stacked():
    """Failure mode stacked bars per benchmark."""
    fm = pd.read_parquet(OUTPUT_DIR / "failure_modes.parquet")
    fm["bench_label"] = fm["seed_source"].map(BENCH_LABELS)

    bench_model_fm = fm.groupby(["seed_source", "model_name"])[
        ["FM1", "FM2", "FM3", "FM4", "FM5"]
    ].mean().reset_index()
    bench_model_fm["model_short"] = bench_model_fm["model_name"].map(MODEL_SHORT)
    bench_model_fm["bench_label"] = bench_model_fm["seed_source"].map(BENCH_LABELS)

    fig, axes = plt.subplots(1, len(bench_model_fm["seed_source"].unique()),
                             figsize=(14, 5), sharey=True)
    if len(bench_model_fm["seed_source"].unique()) == 1:
        axes = [axes]
    colors = sns.color_palette("Set2", 5)

    for idx, bench in enumerate(sorted(bench_model_fm["seed_source"].unique())):
        ax = axes[idx]
        sub = bench_model_fm[bench_model_fm["seed_source"] == bench].sort_values("model_short")
        x = np.arange(len(sub))
        bottom = np.zeros(len(sub))

        for fi, fm_col in enumerate(["FM1", "FM2", "FM3", "FM4", "FM5"]):
            vals = sub[fm_col].values
            ax.bar(x, vals, bottom=bottom, color=colors[fi], label=fm_col if idx == 0 else "")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(sub["model_short"], rotation=30, ha="right", fontsize=9)
        ax.set_title(BENCH_LABELS.get(bench, bench))
        if idx == 0:
            ax.set_ylabel("Cumulative FM Rate")
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Failure Mode Decomposition per Model × Benchmark", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_failure_stacked.png")
    plt.close()
    print("  fig4_failure_stacked.png")


def fig5_commutator_by_axis():
    """Commutator magnitude by demographic axis."""
    cdva = pd.read_parquet(OUTPUT_DIR / "cdva_commutators.parquet")
    if "seed_category" not in cdva.columns or cdva["seed_category"].isna().all():
        print("  fig5 skipped (no seed_category)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    order = cdva.groupby("seed_category")["commutator_mag"].median().sort_values(ascending=False).index
    sns.boxplot(data=cdva, x="seed_category", y="commutator_mag",
                order=order, palette="viridis", ax=ax, showfliers=False)
    ax.set_xlabel("Demographic Axis")
    ax.set_ylabel("|Commutator|")
    ax.set_title("CDVA Commutator Magnitude by Demographic Category")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_commutator_by_axis.png")
    plt.close()
    print("  fig5_commutator_by_axis.png")


def fig6_measdefect_cdf():
    """MeasDefect CDF per model."""
    alg = pd.read_parquet(OUTPUT_DIR / "algebraic_validity.parquet")
    if "meas_defect" not in alg.columns or alg["meas_defect"].isna().all():
        print("  fig6 skipped (no meas_defect)")
        return

    alg["model_short"] = alg["model_name"].map(MODEL_SHORT)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(sorted(alg["model_short"].dropna().unique())):
        sub = alg[alg["model_short"] == model]["meas_defect"].dropna().sort_values()
        cdf = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub.values, cdf, label=model, color=PALETTE[i % len(PALETTE)], linewidth=1.5)

    ax.set_xlabel("MeasDefect (fraction of pairs above τ)")
    ax.set_ylabel("CDF")
    ax.set_title("Measurement Defect Distribution per Model")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_measdefect_cdf.png")
    plt.close()
    print("  fig6_measdefect_cdf.png")


def main():
    print("[06] Generating publication figures...")
    fig1_leaderboard_heatmap()
    fig2_validity_gap_bars()
    fig3_cdva_violin()
    fig4_failure_mode_stacked()
    fig5_commutator_by_axis()
    fig6_measdefect_cdf()
    print(f"\n[06] All figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
