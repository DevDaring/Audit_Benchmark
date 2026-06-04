#!/usr/bin/env python3
"""
Remove all failed (success_flag=False) CDVA rows for nnsight models
(qwen2.5-7b-instruct, phi-4-mini-instruct) and any device-error rows
for gemma-2-2b-it, so the pipeline can re-run CDVA cleanly after the fix.
Llama rows (all success=True) are kept intact.
"""
import sys
from pathlib import Path
import pandas as pd

CDVA_PATH = Path("/home/koushikdeb2009/Audit_Benchmark/Code/mirage/results/cdva_results.parquet")

if not CDVA_PATH.exists():
    print("cdva_results.parquet not found.")
    sys.exit(0)

df = pd.read_parquet(CDVA_PATH)
print(f"Before: {len(df)} rows")
for m, g in df.groupby("model_name"):
    ok = int(g["success_flag"].sum())
    fail = int((~g["success_flag"]).sum())
    print(f"  {m}: success={ok} fail={fail}")

# Keep only Llama successful rows (the one model with clean CDVA)
keep = (df["model_name"] == "llama-3.1-8b-instruct") & (df["success_flag"] == True)
clean = df[keep].copy()
clean.to_parquet(CDVA_PATH, index=False)

removed = len(df) - len(clean)
print(f"\nRemoved {removed} rows (all non-Llama + any failed Llama rows).")
print(f"After: {len(clean)} rows")
for m, g in clean.groupby("model_name"):
    print(f"  {m}: success={int(g['success_flag'].sum())} fail={int((~g['success_flag']).sum())}")
print("CLEAN_OK")
