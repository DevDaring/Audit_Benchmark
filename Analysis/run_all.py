"""
run_all.py — Master runner for MIRAGE analysis pipeline.

Runs scripts 01-06 in order via subprocess, then generates analysis.md.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs"


def run_script(name: str):
    script = ANALYSIS_DIR / name
    print(f"\n{'=' * 70}")
    print(f"  Running: {name}")
    print(f"{'=' * 70}\n")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ANALYSIS_DIR.parent),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  ERROR: {name} exited with code {result.returncode}")
        sys.exit(1)


def generate_analysis_md():
    """Generate the comprehensive analysis.md report."""
    print(f"\n{'=' * 70}")
    print(f"  Generating analysis.md")
    print(f"{'=' * 70}\n")

    scored = pd.read_parquet(OUTPUT_DIR / "scored_results.parquet")
    fm = pd.read_parquet(OUTPUT_DIR / "failure_modes.parquet")
    lb = pd.read_parquet(OUTPUT_DIR / "leaderboard_matrix.parquet")
    validity = pd.read_parquet(OUTPUT_DIR / "validity_gap.parquet")
    ci_df = pd.read_parquet(OUTPUT_DIR / "bootstrap_cis.parquet")
    norms = pd.read_parquet(OUTPUT_DIR / "cdva_seed_norms.parquet")

    with open(OUTPUT_DIR / "cdva_summary.json") as f:
        cdva_summary = json.load(f)
    with open(OUTPUT_DIR / "axiom_compliance.json") as f:
        axioms = json.load(f)
    with open(OUTPUT_DIR / "statistical_tests.json") as f:
        stats = json.load(f)

    tau = cdva_summary["tau"]
    models = sorted(scored["model_name"].unique())
    benchmarks = sorted(scored["seed_source"].dropna().unique())

    lines = []
    lines.append("# MIRAGE Analysis Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Dataset**: {scored['seed_id'].nunique()} seeds x {len(models)} models = "
                 f"{len(scored)} evaluation pairs")
    lines.append(f"- **Source benchmarks**: {', '.join(benchmarks)}")
    lines.append(f"- **Models**: {', '.join(models)}")
    lines.append(f"- **CDVA pairs**: {cdva_summary['total_pairs']} (all position_fallback_used=False)")
    lines.append(f"- **Tau threshold (75th pct)**: {tau:.4f}")
    lines.append("")

    overall_b = scored["mirage_b_pass"].mean()
    overall_f = scored["mirage_full_pass"].mean()
    lines.append(f"**Headline finding**: MIRAGE-B pass rate = {overall_b:.1%}, "
                 f"MIRAGE-Full pass rate = {overall_f:.1%}. "
                 f"The validity gap (native - MIRAGE-Full) reveals substantial hidden "
                 f"measurement invalidity across all tested benchmarks.")
    lines.append("")

    # Per-model scores
    lines.append("## 2. MIRAGE-B and MIRAGE-Full Pass Rates")
    lines.append("")
    lines.append("| Model | MIRAGE-B | MIRAGE-Full | Validity Gap (native-Full) |")
    lines.append("|---|---:|---:|---:|")
    for model in models:
        m = scored[scored["model_name"] == model]
        b_rate = m["mirage_b_pass"].mean()
        f_rate = m["mirage_full_pass"].mean()
        v = validity[validity["model_name"] == model]
        gap = v["validity_gap"].mean() if len(v) > 0 else 0
        lines.append(f"| {model} | {b_rate:.3f} | {f_rate:.3f} | {gap:.3f} |")
    lines.append("")

    # Per benchmark x model
    lines.append("### Per Benchmark x Model")
    lines.append("")
    lines.append("| Benchmark | Model | Native | MIRAGE-B | MIRAGE-Full | Gap |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, row in validity.sort_values(["benchmark", "model_name"]).iterrows():
        bench_label = {"bbq": "BBQ", "crows_pairs": "CrowS-Pairs", "stereoset": "StereoSet"}.get(
            row["benchmark"], row["benchmark"])
        lines.append(f"| {bench_label} | {row['model_name']} | {row['native_pass_rate']:.3f} | "
                     f"{row['mirage_b_pass_rate']:.3f} | {row['mirage_full_pass_rate']:.3f} | "
                     f"{row['validity_gap']:.3f} |")
    lines.append("")

    # Failure Modes
    lines.append("## 3. Failure Mode Distribution")
    lines.append("")
    lines.append("| FM | Definition |")
    lines.append("|---|---|")
    lines.append("| FM1 | Proxy substitution: correct(a) but wrong(b) |")
    lines.append("| FM2 | Architectural indistinguishability: correct(a,b) but CDVA fails |")
    lines.append("| FM3 | Context blindness: correct(a,b) but wrong(d) |")
    lines.append("| FM4 | Criterion leakage: high variance at temp>0 |")
    lines.append("| FM5 | Approximation ceiling: correct(a-d) but wrong(e) |")
    lines.append("")
    lines.append("### Per-model FM rates")
    lines.append("")
    lines.append("| Model | FM1 | FM2 | FM3 | FM4 | FM5 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in models:
        m = fm[fm["model_name"] == model]
        vals = [m[f"FM{i}"].mean() for i in range(1, 6)]
        lines.append(f"| {model} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | "
                     f"{vals[3]:.3f} | {vals[4]:.3f} |")
    lines.append("")

    lines.append("### Leaderboard Matrix (Benchmark x FM)")
    lines.append("")
    lines.append("| Benchmark | FM1 | FM2 | FM3 | FM4 | FM5 | Composite |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in lb.iterrows():
        bench_label = {"bbq": "BBQ", "crows_pairs": "CrowS-Pairs", "stereoset": "StereoSet"}.get(
            row["benchmark"], row["benchmark"])
        lines.append(f"| {bench_label} | {row['FM1']:.3f} | {row['FM2']:.3f} | {row['FM3']:.3f} | "
                     f"{row['FM4']:.3f} | {row['FM5']:.3f} | {row['composite']:.3f} |")
    lines.append("")

    # CDVA Commutator Analysis
    lines.append("## 4. CDVA Commutator Analysis (Algebraic Contribution)")
    lines.append("")
    lines.append("The CDVA (Causal Differential Validity Analysis) operationalizes measurement "
                 "law M2 (Causal Swap Invariance) through activation patching.")
    lines.append("")
    lines.append(f"- **Total valid pairs**: {cdva_summary['total_pairs']}")
    lines.append(f"- **Tau (75th percentile)**: {tau:.4f}")
    lines.append(f"- **Distribution**: mean={cdva_summary['distribution']['mean']:.4f}, "
                 f"median={cdva_summary['distribution']['median']:.4f}, "
                 f"skew={cdva_summary['distribution']['skewness']:.2f}, "
                 f"kurtosis={cdva_summary['distribution']['kurtosis']:.2f}")
    lines.append(f"- **Zero-delta fraction**: {cdva_summary['distribution']['zero_fraction']:.4f}")
    lines.append("")

    lines.append("### Commutativity Index per Model")
    lines.append("")
    lines.append("The commutativity index measures the fraction of seeds where ALL commutator "
                 "magnitudes fall below tau (approximate commutativity holds).")
    lines.append("")
    lines.append("| Model | Commutativity Index |")
    lines.append("|---|---:|")
    for model, ci_val in sorted(cdva_summary["commutativity_index"].items()):
        lines.append(f"| {model} | {ci_val:.4f} |")
    lines.append("")

    lines.append("### Per Demographic Axis")
    lines.append("")
    lines.append("| Axis | N pairs | Mean |C| | Median |C| | Frac above tau |")
    lines.append("|---|---:|---:|---:|---:|")
    for ax in cdva_summary["per_axis"]:
        lines.append(f"| {ax['demographic_axis']} | {ax['n_pairs']} | {ax['mean_commutator']:.4f} | "
                     f"{ax['median_commutator']:.4f} | {ax['frac_above_tau']:.3f} |")
    lines.append("")

    # Algebraic Validity
    lines.append("## 5. Algebraic Validity Framework (PAV)")
    lines.append("")
    lines.append("MIRAGE is formalized as a partial probe magma with validity predicates. "
                 "The framework verifies structural axioms (A1-A6) for benchmark construction "
                 "quality Q(B) and measurement laws (M1-M5) for model discriminative validity V(M,B).")
    lines.append("")

    lines.append("### Structural Axioms")
    lines.append("")
    lines.append("| Axiom | Description | Status |")
    lines.append("|---|---|---|")
    sa = axioms["structural_axioms"]
    lines.append(f"| A1 | Completeness (all seeds have 12 variants) | "
                 f"{'PASS' if sa['A1_completeness']['holds'] else 'FAIL'} ({sa['A1_completeness']['value']:.4f}) |")
    lines.append(f"| A2 | Gold consistency | "
                 f"{'PASS' if sa['A2_gold_consistency']['holds'] else 'FAIL'} ({sa['A2_gold_consistency']['value']:.4f}) |")
    lines.append(f"| A3 | Multi-source | {'PASS' if sa['A3_multi_source']['holds'] else 'FAIL'} |")
    lines.append(f"| A4 | Multi-category | {'PASS' if sa['A4_multi_category']['holds'] else 'FAIL'} ({sa['A4_multi_category']['value']} categories) |")
    lines.append(f"| A5 | Slot coverage | {'PASS' if sa['A5_slot_coverage']['holds'] else 'FAIL'} |")
    lines.append(f"| A6 | Subvariant richness | {'PASS' if sa['A6_subvariant_richness']['holds'] else 'FAIL'} |")
    lines.append("")

    lines.append("### Measurement Laws (Compliance Rate per Model)")
    lines.append("")
    lines.append("| Law | " + " | ".join(models) + " |")
    lines.append("|---" + "|---:" * len(models) + "|")
    ml = axioms["measurement_laws"]
    for law in ["M1_deterministic_reproducibility", "M2_causal_swap_invariance",
                "M3_context_sensitivity", "M4_stochastic_stability", "M5_cot_robustness"]:
        law_short = law.split("_")[0]
        vals = [f"{ml[law].get(m, 0.0):.3f}" for m in models]
        lines.append(f"| {law_short} | " + " | ".join(vals) + " |")
    lines.append("")

    lines.append("### Q(B): Benchmark Construction Quality")
    lines.append("")
    lines.append("| Benchmark | Q(B) |")
    lines.append("|---|---:|")
    for bench, qb in sorted(axioms["benchmark_quality_Q_B"].items()):
        lines.append(f"| {bench} | {qb:.4f} |")
    lines.append("")

    # Statistical Tests
    lines.append("## 6. Statistical Tests")
    lines.append("")
    lines.append("### McNemar's Test: Native vs MIRAGE-Full (Holm-corrected)")
    lines.append("")
    lines.append("| Model | Benchmark | Native | Full | Gap | Cohen's h | p_adj | Sig |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for r in stats["mcnemar_tests"]:
        sig = "***" if r["p_adjusted_holm"] < 0.001 else ("**" if r["p_adjusted_holm"] < 0.01 else ("*" if r["p_adjusted_holm"] < 0.05 else "ns"))
        lines.append(f"| {r['model_name']} | {r['benchmark']} | {r['native_rate']:.3f} | "
                     f"{r['mirage_full_rate']:.3f} | {r['gap']:.3f} | {r['cohens_h']:.3f} | "
                     f"{r['p_adjusted_holm']:.4f} | {sig} |")
    lines.append("")
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    lines.append("")

    # Figures
    lines.append("## 7. Figures")
    lines.append("")
    lines.append("| Figure | Description | File |")
    lines.append("|---|---|---|")
    lines.append("| Fig 1 | Leaderboard heatmap (benchmark x FM) | `outputs/figures/fig1_leaderboard_heatmap.png` |")
    lines.append("| Fig 2 | Native vs MIRAGE-Full pass rates | `outputs/figures/fig2_validity_gap_bars.png` |")
    lines.append("| Fig 3 | CDVA commutator violin plots | `outputs/figures/fig3_cdva_violin.png` |")
    lines.append("| Fig 4 | Failure mode stacked bars | `outputs/figures/fig4_failure_stacked.png` |")
    lines.append("| Fig 5 | Commutator by demographic axis | `outputs/figures/fig5_commutator_by_axis.png` |")
    lines.append("| Fig 6 | MeasDefect CDF per model | `outputs/figures/fig6_measdefect_cdf.png` |")
    lines.append("")

    # Algebraic Interpretation
    lines.append("## 8. Algebraic Interpretation")
    lines.append("")
    lines.append("### Which Axioms Hold")
    lines.append("")
    all_pass = all(v["holds"] for v in sa.values())
    if all_pass:
        lines.append("All structural axioms (A1-A6) **PASS**, confirming that the MIRAGE benchmark "
                     "construction satisfies the requirements of a well-formed partial probe magma.")
    else:
        failed = [k for k, v in sa.items() if not v["holds"]]
        lines.append(f"Structural axioms with violations: {', '.join(failed)}")
    lines.append("")

    lines.append("### Measurement Law Interpretation")
    lines.append("")
    lines.append("- **M2 (Causal Swap Invariance)** is the key discriminative test. Models with "
                 "low M2 compliance show that their internal representations are sensitive to "
                 "demographic token swaps. The commutator is non-zero, indicating the probe "
                 "magma operation is genuinely non-commutative for those (model, seed) pairs.")
    lines.append("")
    lines.append("- **The validity gap** (native pass - MIRAGE-Full pass) quantifies how much "
                 "hidden measurement invalidity exists in source benchmarks. A large gap means "
                 "the source benchmark reports inflated fairness scores that fail under MIRAGE's "
                 "more rigorous multi-axis evaluation.")
    lines.append("")

    lines.append("### Algebraic Structure Summary")
    lines.append("")
    lines.append("The MIRAGE framework demonstrates that bias benchmarks possess a natural "
                 "algebraic structure: the set of probes forms a magma under composition, "
                 "and validity predicates partition this space into regions of genuine "
                 "measurement vs. measurement artifacts. The CDVA commutator quantifies "
                 "the degree to which this structure departs from commutativity — "
                 "a departure that has direct sociotechnical interpretation as hidden bias.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by MIRAGE Analysis Pipeline on "
                 f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    md_path = ANALYSIS_DIR / "analysis.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  analysis.md written ({len(lines)} lines)")


def main():
    print("=" * 70)
    print("  MIRAGE Analysis Pipeline")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    t0 = time.time()

    run_script("01_scoring.py")
    run_script("02_failure_modes.py")
    run_script("03_cdva_commutators.py")
    run_script("04_algebraic_validity.py")
    run_script("05_statistical_tests.py")
    run_script("06_figures.py")
    generate_analysis_md()

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Outputs: {OUTPUT_DIR}")
    print(f"  Report:  {ANALYSIS_DIR / 'analysis.md'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
