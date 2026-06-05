"""
01_scoring.py — Compute MIRAGE-B and MIRAGE-Full per (seed_id x model_name).

Tau calibration: since dev seeds are not tagged, uses the 75th percentile of
|delta_logit| across all CDVA rows as a conservative threshold.

Outputs:
    outputs/scored_results.parquet
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Code" / "mirage"))

RESULTS_DIR = ROOT / "Code" / "mirage" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def answers_match(parsed: str, gold: str) -> bool:
    p = str(parsed).strip().lower()
    g = str(gold).strip().lower()
    if not p or not g:
        return False
    return p == g or g in p or p in g


def majority_vote(answers: pd.Series) -> str | None:
    if answers.empty:
        return None
    vc = answers.value_counts()
    if vc.iloc[0] > len(answers) / 2:
        return vc.index[0]
    return None


def compute_mirage_b(rows: pd.DataFrame, gold: str) -> bool:
    """MIRAGE-B pass for a single (seed_id, model_name) group."""
    det = rows[rows["sample_index"] == 0]

    def slot_correct(slot, subvariant):
        r = det[(det["slot"] == slot) & (det["subvariant"] == subvariant) & (det["success_flag"] == True)]
        if r.empty:
            return False
        return answers_match(str(r.iloc[0]["parsed_answer"]), gold)

    correct_a = slot_correct("a", "surface")
    correct_b = slot_correct("b", "iso_control")

    c_rows = det[(det["slot"] == "c") & (det["success_flag"] == True)]
    if len(c_rows) >= 3:
        distinct_c = c_rows.drop_duplicates(subset=["prompt_text"])
        mv = majority_vote(distinct_c["parsed_answer"])
        stable_c = mv is not None and answers_match(mv, gold)
    else:
        stable_c = False

    correct_d = slot_correct("d", "d_valid") and slot_correct("d", "d_harmful")

    e_rows = det[(det["slot"] == "e") & (det["success_flag"] == True)]
    if len(e_rows) >= 2:
        mv_e = majority_vote(e_rows["parsed_answer"])
        cot_robust = mv_e is not None and answers_match(mv_e, gold)
    else:
        cot_robust = False

    return all([correct_a, correct_b, stable_c, correct_d, cot_robust])


def calibrate_tau(cdva_df: pd.DataFrame) -> float:
    """75th percentile of |delta_logit| as conservative tau threshold."""
    valid = cdva_df[cdva_df["success_flag"] == True]
    if valid.empty:
        return 0.5
    return float(np.percentile(np.abs(valid["delta_logit"].values), 75))


def main():
    print("[01] Loading data...")
    beh = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet")
    cdva = pd.read_parquet(RESULTS_DIR / "cdva_results.parquet")
    cdva = cdva[cdva["position_fallback_used"] == False]

    tau = calibrate_tau(cdva)
    print(f"[01] Calibrated tau (75th pct |delta_logit|): {tau:.4f}")

    seed_models = beh.groupby(["seed_id", "model_name"]).first().reset_index()[
        ["seed_id", "model_name", "seed_source", "seed_category", "gold_answer"]
    ]

    records = []
    total = len(seed_models)
    for idx, (_, row) in enumerate(seed_models.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"  Scoring {idx+1}/{total}...")
        sid, model = row["seed_id"], row["model_name"]
        gold = str(row["gold_answer"]) if pd.notna(row["gold_answer"]) else ""

        model_rows = beh[(beh["seed_id"] == sid) & (beh["model_name"] == model)]
        b_pass = compute_mirage_b(model_rows, gold)

        f_pass = False
        if b_pass:
            seed_cdva = cdva[(cdva["seed_id"] == sid) & (cdva["model_name"] == model) & (cdva["success_flag"] == True)]
            if not seed_cdva.empty:
                cdva_score = float(seed_cdva["cdva_pair_score"].mean())
                f_pass = cdva_score > tau

        records.append({
            "seed_id": sid,
            "model_name": model,
            "seed_source": row["seed_source"],
            "seed_category": row["seed_category"],
            "mirage_b_pass": b_pass,
            "mirage_full_pass": f_pass,
            "tau": tau,
        })

    scored = pd.DataFrame(records)
    out_path = OUTPUT_DIR / "scored_results.parquet"
    scored.to_parquet(out_path, index=False)

    print(f"\n[01] Results: {len(scored)} (seed x model) pairs scored")
    print(f"  MIRAGE-B pass rate:    {scored['mirage_b_pass'].mean():.4f}")
    print(f"  MIRAGE-Full pass rate: {scored['mirage_full_pass'].mean():.4f}")
    for model in sorted(scored["model_name"].unique()):
        m = scored[scored["model_name"] == model]
        print(f"  {model:30s} B={m['mirage_b_pass'].mean():.3f}  Full={m['mirage_full_pass'].mean():.3f}")
    print(f"  Saved: {out_path}")
    return scored


if __name__ == "__main__":
    main()
