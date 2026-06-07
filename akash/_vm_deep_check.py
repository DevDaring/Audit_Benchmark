#!/usr/bin/env python3
"""Deep result integrity check for journal-quality validation."""
import sys
from pathlib import Path
import pandas as pd

MIRAGE = Path("/home/koushikdeb2009/Audit_Benchmark/Code/mirage")
sys.path.insert(0, str(MIRAGE))
from config import RESULTS_DIR, SEEDS_DIR

beh_path = RESULTS_DIR / "behavioral_results.parquet"
cdva_path = RESULTS_DIR / "cdva_results.parquet"
pentad_path = SEEDS_DIR / "pentad_dataset.parquet"

# ── BEHAVIORAL ──────────────────────────────────────────────────────────────
print("=== BEHAVIORAL ===")
if beh_path.exists():
    b = pd.read_parquet(beh_path)
    print(f"total_rows: {len(b)}")
    for m, grp in b.groupby("model_name"):
        det = grp[grp["sample_index"] == 0]
        var = grp[grp["sample_index"] > 0]
        ok = int(grp["success_flag"].sum())
        fail = int((~grp["success_flag"]).sum())
        print(f"  {m}")
        print(f"    det_rows={len(det)}  var_rows={len(var)}  success={ok}  fail={fail}")
        print(f"    det_seeds={det['seed_id'].nunique()}  det_slots={dict(det['slot'].value_counts())}")
        si_counts = dict(grp.groupby("sample_index").size())
        print(f"    sample_index_counts={si_counts}")
else:
    print("MISSING")

# ── CDVA ────────────────────────────────────────────────────────────────────
print("\n=== CDVA ===")
if cdva_path.exists():
    c = pd.read_parquet(cdva_path)
    print(f"total_rows: {len(c)}")
    for m, grp in c.groupby("model_name"):
        ok = int(grp["success_flag"].sum())
        fail = int((~grp["success_flag"]).sum())
        seeds_ok = int(grp[grp["success_flag"]]["seed_id"].nunique())
        fr = grp[~grp["success_flag"]]["failure_reason"].value_counts().to_dict() if fail > 0 else {}
        print(f"  {m}: rows={len(grp)} success={ok} fail={fail} seeds_with_success={seeds_ok}")
        if fr:
            print(f"    failure_reasons={fr}")
else:
    print("MISSING")

# ── PENTAD SANITY ───────────────────────────────────────────────────────────
print("\n=== PENTAD ===")
if pentad_path.exists():
    p = pd.read_parquet(pentad_path)
    print(f"rows={len(p)} seeds={p['seed_id'].nunique()} slots={dict(p['slot'].value_counts())}")
    null_prompts = int(p["prompt_text"].isna().sum())
    print(f"null_prompts={null_prompts}")
else:
    print("MISSING")
