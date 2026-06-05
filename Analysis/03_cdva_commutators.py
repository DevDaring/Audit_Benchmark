"""
03_cdva_commutators.py — Algebraic CDVA commutator analysis.

Computes:
- Commutator magnitude |C(M, s, i, j)| = |delta_logit| per pair
- Seed-level commutator norm (L2 norm of the commutator vector per seed)
- Model commutativity index (fraction of seeds where all |C| < tau)
- Asymmetry detection: C(i->j) vs C(j->i) direction dependence
- Per-axis (demographic) commutator statistics
- Structural defect spectrum (distribution characteristics)

Outputs:
    outputs/cdva_commutators.parquet     (enriched per-pair)
    outputs/cdva_seed_norms.parquet      (per seed x model)
    outputs/cdva_summary.json            (aggregate statistics)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Code" / "mirage" / "results"
DATASET_DIR = ROOT / "Code" / "mirage" / "Dataset" / "seeds"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("[03] Loading CDVA data...")
    cdva = pd.read_parquet(RESULTS_DIR / "cdva_results.parquet")
    cdva = cdva[cdva["position_fallback_used"] == False].copy()
    cdva = cdva[cdva["success_flag"] == True].copy()

    pentad = pd.read_parquet(DATASET_DIR / "pentad_dataset.parquet")
    seed_meta = pentad[["seed_id", "seed_source", "seed_category"]].drop_duplicates("seed_id")
    cdva = cdva.merge(seed_meta, on="seed_id", how="left")

    # Commutator magnitude
    cdva["commutator_mag"] = cdva["delta_logit"].abs()

    # Tau: 75th percentile
    tau = float(np.percentile(cdva["commutator_mag"].values, 75))
    print(f"[03] Tau (75th pct): {tau:.4f}")

    # Save enriched per-pair data
    cdva.to_parquet(OUTPUT_DIR / "cdva_commutators.parquet", index=False)

    # --- Seed-level norms ---
    seed_norms = []
    for (sid, model), grp in cdva.groupby(["seed_id", "model_name"]):
        mags = grp["commutator_mag"].values
        l2_norm = float(np.linalg.norm(mags))
        l1_norm = float(np.sum(mags))
        max_mag = float(np.max(mags))
        mean_mag = float(np.mean(mags))
        n_pairs = len(mags)
        all_below_tau = int(np.all(mags < tau))

        seed_norms.append({
            "seed_id": sid,
            "model_name": model,
            "seed_source": grp["seed_source"].iloc[0] if "seed_source" in grp.columns else "",
            "seed_category": grp["seed_category"].iloc[0] if "seed_category" in grp.columns else "",
            "n_pairs": n_pairs,
            "commutator_l2_norm": l2_norm,
            "commutator_l1_norm": l1_norm,
            "commutator_max": max_mag,
            "commutator_mean": mean_mag,
            "all_below_tau": all_below_tau,
        })

    norms_df = pd.DataFrame(seed_norms)
    norms_df.to_parquet(OUTPUT_DIR / "cdva_seed_norms.parquet", index=False)

    # --- Commutativity index per model ---
    commutativity = {}
    for model in cdva["model_name"].unique():
        m_norms = norms_df[norms_df["model_name"] == model]
        commutativity[model] = float(m_norms["all_below_tau"].mean())

    # --- Asymmetry analysis ---
    # Group pairs by (seed, model) and check if A->B differs from B->A
    asymmetry_records = []
    pair_groups = cdva.groupby(["seed_id", "model_name"])
    for (sid, model), grp in pair_groups:
        pairs = grp[["pair_A_subvariant", "pair_B_subvariant", "delta_logit"]].values
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                if pairs[i][0] == pairs[j][1] and pairs[i][1] == pairs[j][0]:
                    asymmetry_records.append({
                        "seed_id": sid,
                        "model_name": model,
                        "pair": f"{pairs[i][0]}<->{pairs[i][1]}",
                        "forward_delta": float(pairs[i][2]),
                        "reverse_delta": float(pairs[j][2]),
                        "asymmetry": abs(float(pairs[i][2]) - float(pairs[j][2])),
                    })

    asym_df = pd.DataFrame(asymmetry_records) if asymmetry_records else pd.DataFrame()

    # --- Per-axis statistics ---
    axis_stats = []
    if "seed_category" in cdva.columns:
        for cat in sorted(cdva["seed_category"].dropna().unique()):
            cat_data = cdva[cdva["seed_category"] == cat]
            mags = cat_data["commutator_mag"].values
            axis_stats.append({
                "demographic_axis": cat,
                "n_pairs": len(mags),
                "mean_commutator": float(np.mean(mags)),
                "median_commutator": float(np.median(mags)),
                "std_commutator": float(np.std(mags)),
                "max_commutator": float(np.max(mags)),
                "frac_above_tau": float(np.mean(mags >= tau)),
            })
    axis_df = pd.DataFrame(axis_stats)

    # --- Distribution characteristics ---
    all_mags = cdva["commutator_mag"].values
    skewness = float(sp_stats.skew(all_mags))
    kurtosis = float(sp_stats.kurtosis(all_mags))
    zero_frac = float(np.mean(all_mags == 0))
    ks_stat, ks_p = sp_stats.kstest(all_mags[all_mags > 0], "expon",
                                     args=(0, np.mean(all_mags[all_mags > 0])))

    # --- Summary ---
    summary = {
        "tau": tau,
        "total_pairs": int(len(cdva)),
        "total_seeds": int(cdva["seed_id"].nunique()),
        "models": sorted(cdva["model_name"].unique().tolist()),
        "commutativity_index": commutativity,
        "distribution": {
            "mean": float(np.mean(all_mags)),
            "median": float(np.median(all_mags)),
            "std": float(np.std(all_mags)),
            "skewness": skewness,
            "kurtosis": kurtosis,
            "zero_fraction": zero_frac,
            "ks_exponential_test": {"statistic": float(ks_stat), "p_value": float(ks_p)},
        },
        "asymmetry": {
            "n_reciprocal_pairs": len(asym_df),
            "mean_asymmetry": float(asym_df["asymmetry"].mean()) if len(asym_df) > 0 else 0.0,
            "max_asymmetry": float(asym_df["asymmetry"].max()) if len(asym_df) > 0 else 0.0,
        },
        "per_axis": axis_stats,
    }

    with open(OUTPUT_DIR / "cdva_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[03] CDVA Commutator Analysis ({len(cdva)} valid pairs):")
    print(f"  Tau (75th percentile): {tau:.4f}")
    print(f"  Distribution: mean={np.mean(all_mags):.4f}, median={np.median(all_mags):.4f}, "
          f"skew={skewness:.2f}, kurtosis={kurtosis:.2f}")
    print(f"  Zero-delta fraction: {zero_frac:.4f}")
    print(f"\n  Commutativity index (all |C| < tau for seed):")
    for model, ci in sorted(commutativity.items()):
        print(f"    {model:30s} {ci:.4f}")
    if len(asym_df) > 0:
        print(f"\n  Asymmetry: {len(asym_df)} reciprocal pairs, "
              f"mean={asym_df['asymmetry'].mean():.4f}")
    print(f"\n  Per demographic axis:")
    for row in axis_stats:
        print(f"    {row['demographic_axis']:20s} mean={row['mean_commutator']:.4f} "
              f"above_tau={row['frac_above_tau']:.3f}")
    print(f"\n  Saved: outputs/cdva_commutators.parquet, cdva_seed_norms.parquet, cdva_summary.json")
    return summary


if __name__ == "__main__":
    main()
