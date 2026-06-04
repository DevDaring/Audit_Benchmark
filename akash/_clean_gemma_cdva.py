#!/usr/bin/env python3
"""
Remove all gemma-2-2b-it rows from cdva_results.parquet so the pipeline
can re-run Gemma CDVA cleanly after the device-mismatch fix.
Keeps Llama and Qwen rows intact.
"""
import sys
from pathlib import Path

RESULTS = Path("/home/koushikdeb2009/Audit_Benchmark/Code/mirage/results")
CDVA_PATH = RESULTS / "cdva_results.parquet"

import pandas as pd

if not CDVA_PATH.exists():
    print("cdva_results.parquet not found — nothing to clean.")
    sys.exit(0)

df = pd.read_parquet(CDVA_PATH)
print(f"Before: {len(df)} rows, models: {dict(df['model_name'].value_counts())}")

gemma_mask = df["model_name"] == "gemma-2-2b-it"
n_removed = gemma_mask.sum()

clean = df[~gemma_mask].copy()
clean.to_parquet(CDVA_PATH, index=False)

print(f"Removed {n_removed} gemma-2-2b-it rows (all had success_flag=False due to device bug).")
print(f"After:  {len(clean)} rows, models: {dict(clean['model_name'].value_counts())}")
print("CLEAN_OK")
