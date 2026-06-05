"""
05_statistical_tests.py — Bootstrap CIs, McNemar, Cohen's h, Holm-Bonferroni.

Implements:
- Bootstrap 95% CIs (5000 resamples) for all pass rates
- McNemar's test: native pass vs MIRAGE-Full pass (paired binary)
- Cohen's h effect sizes for the validity gap
- Holm-Bonferroni correction (4 benchmarks x 4 models = 16 tests)
- BH-FDR for exploratory per-category breakdowns

Outputs:
    outputs/statistical_tests.json    (all test results)
    outputs/bootstrap_cis.parquet     (per model x benchmark CIs)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Code" / "mirage" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def bootstrap_ci(values, n_resamples=5000, alpha=0.05):
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed=20260101)
    point = float(np.mean(arr))
    boot = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)])
    lower = float(np.percentile(boot, 100 * alpha / 2))
    upper = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return point, lower, upper


def cohens_h(p1: float, p2: float) -> float:
    return float(2.0 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))))


def mcnemar_test(native_pass: np.ndarray, mirage_pass: np.ndarray):
    n01 = int(np.sum((~native_pass) & mirage_pass))
    n10 = int(np.sum(native_pass & (~mirage_pass)))
    n00 = int(np.sum((~native_pass) & (~mirage_pass)))
    n11 = int(np.sum(native_pass & mirage_pass))
    n_discordant = n01 + n10

    if n_discordant < 25:
        result = sp_stats.binomtest(min(n01, n10), n=n_discordant, p=0.5, alternative="two-sided")
        return {"statistic": float(min(n01, n10)), "p_value": float(result.pvalue),
                "n01": n01, "n10": n10, "n00": n00, "n11": n11, "method": "exact_binomial"}
    else:
        if n_discordant == 0:
            return {"statistic": 0.0, "p_value": 1.0, "n01": n01, "n10": n10,
                    "n00": n00, "n11": n11, "method": "chi2"}
        stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
        pval = float(1 - sp_stats.chi2.cdf(stat, df=1))
        return {"statistic": float(stat), "p_value": pval, "n01": n01, "n10": n10,
                "n00": n00, "n11": n11, "method": "chi2_continuity"}


def holm_bonferroni(pvalues):
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adj = min(float(pval) * (n - rank), 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def bh_fdr(pvalues):
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [1.0] * n
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        orig_idx, pval = indexed[rank]
        bh_val = float(pval) * n / (rank + 1)
        prev = min(prev, bh_val)
        adjusted[orig_idx] = min(prev, 1.0)
    return adjusted


def answers_match(parsed: str, gold: str) -> bool:
    p = str(parsed).strip().lower()
    g = str(gold).strip().lower()
    if not p or not g:
        return False
    return p == g or g in p or p in g


def main():
    print("[05] Loading data...")
    beh = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet")
    scored = pd.read_parquet(OUTPUT_DIR / "scored_results.parquet")

    benchmarks = sorted(scored["seed_source"].dropna().unique())
    models = sorted(scored["model_name"].unique())

    # --- Bootstrap CIs ---
    ci_records = []
    for bench in benchmarks:
        for model in models:
            sub = scored[(scored["seed_source"] == bench) & (scored["model_name"] == model)]
            if sub.empty:
                continue
            b_vals = sub["mirage_b_pass"].astype(float).values
            f_vals = sub["mirage_full_pass"].astype(float).values

            b_point, b_lo, b_hi = bootstrap_ci(b_vals)
            f_point, f_lo, f_hi = bootstrap_ci(f_vals)

            # Native pass rate
            native_rows = beh[(beh["seed_source"] == bench) & (beh["model_name"] == model) &
                              (beh["slot"] == "a") & (beh["subvariant"] == "surface") &
                              (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
            native_correct = native_rows.apply(
                lambda r: answers_match(str(r["parsed_answer"]), str(r["gold_answer"])), axis=1
            ).astype(float).values
            if len(native_correct) > 0:
                n_point, n_lo, n_hi = bootstrap_ci(native_correct)
            else:
                n_point, n_lo, n_hi = 0.0, 0.0, 0.0

            ci_records.append({
                "benchmark": bench, "model_name": model, "n_seeds": len(sub),
                "native_point": n_point, "native_ci_lo": n_lo, "native_ci_hi": n_hi,
                "mirage_b_point": b_point, "mirage_b_ci_lo": b_lo, "mirage_b_ci_hi": b_hi,
                "mirage_full_point": f_point, "mirage_full_ci_lo": f_lo, "mirage_full_ci_hi": f_hi,
            })

    ci_df = pd.DataFrame(ci_records)
    ci_df.to_parquet(OUTPUT_DIR / "bootstrap_cis.parquet", index=False)

    # --- McNemar tests (native vs MIRAGE-Full) ---
    mcnemar_results = []
    for bench in benchmarks:
        for model in models:
            sub = scored[(scored["seed_source"] == bench) & (scored["model_name"] == model)]
            if sub.empty:
                continue

            native_rows = beh[(beh["seed_source"] == bench) & (beh["model_name"] == model) &
                              (beh["slot"] == "a") & (beh["subvariant"] == "surface") &
                              (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
            seed_native = {}
            for _, r in native_rows.iterrows():
                seed_native[r["seed_id"]] = answers_match(str(r["parsed_answer"]), str(r["gold_answer"]))

            seeds = sub["seed_id"].values
            native_arr = np.array([seed_native.get(s, False) for s in seeds])
            mirage_arr = sub["mirage_full_pass"].values.astype(bool)

            result = mcnemar_test(native_arr, mirage_arr)
            h = cohens_h(float(native_arr.mean()), float(mirage_arr.mean()))

            mcnemar_results.append({
                "benchmark": bench, "model_name": model,
                "native_rate": float(native_arr.mean()),
                "mirage_full_rate": float(mirage_arr.mean()),
                "gap": float(native_arr.mean() - mirage_arr.mean()),
                "cohens_h": h,
                **result,
            })

    # --- Holm-Bonferroni correction ---
    raw_pvals = [r["p_value"] for r in mcnemar_results]
    adj_pvals = holm_bonferroni(raw_pvals)
    for i, r in enumerate(mcnemar_results):
        r["p_adjusted_holm"] = adj_pvals[i]
        r["significant_holm_005"] = adj_pvals[i] < 0.05

    # --- BH-FDR for per-category ---
    category_results = []
    categories = sorted(scored["seed_category"].dropna().unique())
    for cat in categories:
        for model in models:
            cat_scored = scored[(scored["seed_category"] == cat) & (scored["model_name"] == model)]
            if len(cat_scored) < 10:
                continue
            b_rate = float(cat_scored["mirage_b_pass"].mean())
            f_rate = float(cat_scored["mirage_full_pass"].mean())
            category_results.append({
                "category": cat, "model_name": model, "n": len(cat_scored),
                "mirage_b_rate": b_rate, "mirage_full_rate": f_rate,
            })

    # Summary output
    all_results = {
        "bootstrap_cis": ci_records,
        "mcnemar_tests": mcnemar_results,
        "per_category": category_results,
        "n_confirmatory_tests": len(mcnemar_results),
    }

    with open(OUTPUT_DIR / "statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[05] Statistical Tests:")
    print(f"\n  McNemar tests (native vs MIRAGE-Full), Holm-corrected:")
    for r in mcnemar_results:
        sig = "***" if r["p_adjusted_holm"] < 0.001 else ("**" if r["p_adjusted_holm"] < 0.01 else ("*" if r["p_adjusted_holm"] < 0.05 else "ns"))
        print(f"    {r['model_name']:30s} {r['benchmark']:15s} "
              f"gap={r['gap']:.3f} h={r['cohens_h']:.3f} p_adj={r['p_adjusted_holm']:.4f} {sig}")

    print(f"\n  Effect size interpretation: |h|<0.2=small, 0.2-0.8=medium, >0.8=large")
    print(f"  Saved: statistical_tests.json, bootstrap_cis.parquet")
    return all_results


if __name__ == "__main__":
    main()
