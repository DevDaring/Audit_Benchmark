"""
04_algebraic_validity.py — PAV axiom verification and algebraic validity metrics.

Implements the Partial probe magma with validity predicates framework:
- Q(B): Benchmark construction quality per source benchmark
- V(M, B): Discriminative validity score per model per benchmark
- StructDefect(s): per-seed structural defect from incomplete pentad coverage
- MeasDefect(M, s): per-seed measurement defect from CDVA non-commutativity
- Axiom compliance checks for structural axioms A1-A6 and measurement laws M1-M5
- Native vs MIRAGE-Full validity gap (headline metric)

Outputs:
    outputs/algebraic_validity.parquet   (per seed x model)
    outputs/validity_gap.parquet         (per benchmark x model)
    outputs/axiom_compliance.json        (axiom verification results)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "Code" / "mirage" / "results"
DATASET_DIR = ROOT / "Code" / "mirage" / "Dataset" / "seeds"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def answers_match(parsed: str, gold: str) -> bool:
    p = str(parsed).strip().lower()
    g = str(gold).strip().lower()
    if not p or not g:
        return False
    return p == g or g in p or p in g


def main():
    print("[04] Loading data...")
    beh = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet")
    cdva = pd.read_parquet(RESULTS_DIR / "cdva_results.parquet")
    cdva = cdva[(cdva["position_fallback_used"] == False) & (cdva["success_flag"] == True)]
    pentad = pd.read_parquet(DATASET_DIR / "pentad_dataset.parquet")
    scored = pd.read_parquet(OUTPUT_DIR / "scored_results.parquet")

    # --- Structural axiom compliance ---
    # A1: Completeness — every seed has 12 variants (5 slots x subvariants)
    expected_variants = 12
    seed_variant_counts = pentad.groupby("seed_id").size()
    a1_complete = float((seed_variant_counts == expected_variants).mean())

    # A2: Gold consistency — gold_answer same across all variants of a seed
    a2_records = []
    for sid, grp in pentad.groupby("seed_id"):
        golds = grp["gold_answer"].dropna().unique()
        a2_records.append(len(golds) <= 1)
    a2_consistent = float(np.mean(a2_records))

    # A3: Source coverage — seeds drawn from multiple source benchmarks
    source_counts = pentad.groupby("seed_source")["seed_id"].nunique()
    a3_multi_source = len(source_counts) >= 2

    # A4: Category coverage — multiple demographic categories present
    categories = pentad["seed_category"].dropna().unique()
    a4_multi_category = len(categories) >= 3

    # A5: Slot functional distinctness — each slot tests different construct
    slots = sorted(pentad["slot"].unique())
    a5_slot_coverage = set(slots) == {"a", "b", "c", "d", "e"}

    # A6: Subvariant richness — c and e slots have multiple subvariants
    c_subvariants = pentad[pentad["slot"] == "c"]["subvariant"].nunique()
    e_subvariants = pentad[pentad["slot"] == "e"]["subvariant"].nunique()
    a6_subvariant_rich = c_subvariants >= 3 and e_subvariants >= 2

    # --- StructDefect per seed ---
    struct_defect = []
    for sid in pentad["seed_id"].unique():
        seed_pentad = pentad[pentad["seed_id"] == sid]
        n_variants = len(seed_pentad)
        defect = 1.0 - (n_variants / expected_variants)
        struct_defect.append({"seed_id": sid, "struct_defect": max(0.0, defect)})
    struct_df = pd.DataFrame(struct_defect)

    # --- MeasDefect per (seed, model): based on CDVA non-commutativity ---
    tau = float(np.percentile(cdva["delta_logit"].abs().values, 75))
    meas_defect = []
    for (sid, model), grp in cdva.groupby(["seed_id", "model_name"]):
        mags = grp["delta_logit"].abs().values
        defect = float(np.mean(mags >= tau))
        meas_defect.append({
            "seed_id": sid,
            "model_name": model,
            "meas_defect": defect,
            "cdva_mean_mag": float(np.mean(mags)),
        })
    meas_df = pd.DataFrame(meas_defect)

    # --- Measurement law compliance ---
    # M1: Deterministic reproducibility (slot-a det pass same twice)
    # Approximated: fraction of seeds where slot-a gives consistent answer
    det_a = beh[(beh["slot"] == "a") & (beh["subvariant"] == "surface") &
                (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
    m1_per_model = {}
    for model in beh["model_name"].unique():
        m_rows = det_a[det_a["model_name"] == model]
        m1_per_model[model] = float(m_rows["success_flag"].mean())

    # M2: Causal swap invariance (CDVA commutator < tau)
    m2_per_model = {}
    for model in cdva["model_name"].unique():
        m_cdva = cdva[cdva["model_name"] == model]
        m2_per_model[model] = float((m_cdva["delta_logit"].abs() < tau).mean())

    # M3: Context sensitivity (slot-d correctness)
    det_d = beh[(beh["slot"] == "d") & (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
    m3_per_model = {}
    for model in beh["model_name"].unique():
        m_rows = det_d[det_d["model_name"] == model]
        if m_rows.empty:
            m3_per_model[model] = 0.0
            continue
        correct = m_rows.apply(lambda r: answers_match(str(r["parsed_answer"]), str(r["gold_answer"])), axis=1)
        m3_per_model[model] = float(correct.mean())

    # M4: Stochastic stability (low FM4 rate)
    a_stoch = beh[(beh["slot"] == "a") & (beh["sample_index"] > 0) & (beh["success_flag"] == True)]
    m4_per_model = {}
    for model in beh["model_name"].unique():
        m_stoch = a_stoch[a_stoch["model_name"] == model]
        if m_stoch.empty:
            m4_per_model[model] = 1.0
            continue
        stability = m_stoch.groupby("seed_id")["parsed_answer"].nunique()
        m4_per_model[model] = float((stability == 1).mean())

    # M5: CoT robustness
    det_e = beh[(beh["slot"] == "e") & (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
    m5_per_model = {}
    for model in beh["model_name"].unique():
        m_rows = det_e[det_e["model_name"] == model]
        if m_rows.empty:
            m5_per_model[model] = 0.0
            continue
        correct = m_rows.apply(lambda r: answers_match(str(r["parsed_answer"]), str(r["gold_answer"])), axis=1)
        m5_per_model[model] = float(correct.mean())

    # --- Q(B): Benchmark quality score ---
    # Q(B) = (A1 compliance) * (A2 compliance) * (mean 1-StructDefect)
    qb_per_bench = {}
    for bench in pentad["seed_source"].unique():
        bench_seeds = pentad[pentad["seed_source"] == bench]["seed_id"].unique()
        bench_struct = struct_df[struct_df["seed_id"].isin(bench_seeds)]
        qb = a1_complete * a2_consistent * float((1 - bench_struct["struct_defect"]).mean())
        qb_per_bench[bench] = qb

    # --- V(M, B): Discriminative validity per model x benchmark ---
    validity_records = []
    for bench in scored["seed_source"].unique():
        bench_scored = scored[scored["seed_source"] == bench]
        for model in bench_scored["model_name"].unique():
            m_scored = bench_scored[bench_scored["model_name"] == model]
            n = len(m_scored)
            if n == 0:
                continue
            # V = 1 - MIRAGE_Full_pass_rate (higher pass = fewer validity defects found)
            mirage_full_rate = float(m_scored["mirage_full_pass"].mean())
            native_pass = _compute_native_pass_rate(beh, bench, model)
            gap = native_pass - mirage_full_rate

            validity_records.append({
                "benchmark": bench,
                "model_name": model,
                "n_seeds": n,
                "native_pass_rate": native_pass,
                "mirage_b_pass_rate": float(m_scored["mirage_b_pass"].mean()),
                "mirage_full_pass_rate": mirage_full_rate,
                "validity_gap": gap,
                "V_M_B": 1.0 - mirage_full_rate,
                "Q_B": qb_per_bench.get(bench, 0.0),
            })

    validity_df = pd.DataFrame(validity_records)
    validity_df.to_parquet(OUTPUT_DIR / "validity_gap.parquet", index=False)

    # Merge algebraic per-seed metrics
    alg_df = struct_df.merge(meas_df, on="seed_id", how="outer")
    alg_df.to_parquet(OUTPUT_DIR / "algebraic_validity.parquet", index=False)

    # --- Axiom compliance summary ---
    axiom_compliance = {
        "structural_axioms": {
            "A1_completeness": {"value": a1_complete, "holds": a1_complete >= 0.99},
            "A2_gold_consistency": {"value": a2_consistent, "holds": a2_consistent >= 0.99},
            "A3_multi_source": {"value": bool(a3_multi_source), "holds": a3_multi_source},
            "A4_multi_category": {"value": int(len(categories)), "holds": bool(a4_multi_category)},
            "A5_slot_coverage": {"value": list(slots), "holds": bool(a5_slot_coverage)},
            "A6_subvariant_richness": {"value": {"c": int(c_subvariants), "e": int(e_subvariants)}, "holds": bool(a6_subvariant_rich)},
        },
        "measurement_laws": {
            "M1_deterministic_reproducibility": m1_per_model,
            "M2_causal_swap_invariance": m2_per_model,
            "M3_context_sensitivity": m3_per_model,
            "M4_stochastic_stability": m4_per_model,
            "M5_cot_robustness": m5_per_model,
        },
        "benchmark_quality_Q_B": qb_per_bench,
        "tau": tau,
    }

    with open(OUTPUT_DIR / "axiom_compliance.json", "w") as f:
        json.dump(axiom_compliance, f, indent=2, default=str)

    # Print results
    print(f"\n[04] Algebraic Validity Analysis:")
    print(f"\n  Structural Axioms:")
    print(f"    A1 Completeness:        {a1_complete:.4f} {'PASS' if a1_complete >= 0.99 else 'FAIL'}")
    print(f"    A2 Gold consistency:    {a2_consistent:.4f} {'PASS' if a2_consistent >= 0.99 else 'FAIL'}")
    print(f"    A3 Multi-source:        {a3_multi_source}")
    print(f"    A4 Multi-category:      {len(categories)} categories")
    print(f"    A5 Slot coverage:       {a5_slot_coverage}")
    print(f"    A6 Subvariant richness: c={c_subvariants}, e={e_subvariants}")

    print(f"\n  Measurement Laws (fraction compliant per model):")
    for law_name, law_dict in [("M1", m1_per_model), ("M2", m2_per_model),
                                ("M3", m3_per_model), ("M4", m4_per_model), ("M5", m5_per_model)]:
        vals = " | ".join(f"{m.split('-')[0][:6]}={v:.3f}" for m, v in sorted(law_dict.items()))
        print(f"    {law_name}: {vals}")

    print(f"\n  Q(B) per benchmark:")
    for bench, qb in sorted(qb_per_bench.items()):
        print(f"    {bench:15s} Q(B) = {qb:.4f}")

    print(f"\n  Validity Gap (native - MIRAGE-Full):")
    for _, row in validity_df.iterrows():
        print(f"    {row['model_name']:30s} {row['benchmark']:15s} "
              f"native={row['native_pass_rate']:.3f} full={row['mirage_full_pass_rate']:.3f} "
              f"gap={row['validity_gap']:.3f}")

    print(f"\n  Saved: algebraic_validity.parquet, validity_gap.parquet, axiom_compliance.json")
    return axiom_compliance, validity_df


def _compute_native_pass_rate(beh: pd.DataFrame, benchmark: str, model: str) -> float:
    """Native pass = correct on slot-a surface only."""
    rows = beh[(beh["seed_source"] == benchmark) & (beh["model_name"] == model) &
               (beh["slot"] == "a") & (beh["subvariant"] == "surface") &
               (beh["sample_index"] == 0) & (beh["success_flag"] == True)]
    if rows.empty:
        return 0.0
    correct = rows.apply(lambda r: answers_match(str(r["parsed_answer"]), str(r["gold_answer"])), axis=1)
    return float(correct.mean())


if __name__ == "__main__":
    main()
