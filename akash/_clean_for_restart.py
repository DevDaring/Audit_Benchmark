#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

p = Path("/home/koushikdeb2009/Audit_Benchmark/Code/mirage/results/cdva_results.parquet")
df = pd.read_parquet(p)
before = len(df)
clean = df[(df["model_name"] == "llama-3.1-8b-instruct") & (df["success_flag"] == True)].copy()
clean.to_parquet(p, index=False)
print(f"Kept {len(clean)} Llama rows, removed {before - len(clean)}")
print("CLEAN_OK")
