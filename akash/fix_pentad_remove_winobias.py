"""One-time fix: remove WinoBias from pentad_dataset.parquet (held-out benchmark)."""
import sys
from pathlib import Path

sys.path.insert(0, "/home/koushikdeb2009/Audit_Benchmark/Code/mirage")

import pandas as pd
from config import SEEDS_DIR
from Dataset.validate_pentad import assert_production_ready

path = SEEDS_DIR / "pentad_dataset.parquet"
df = pd.read_parquet(path)
n_before = len(df)
n_wino = int((df["seed_source"].astype(str).str.lower() == "winobias").sum())

if n_wino == 0:
    print("No WinoBias rows — already clean.")
    assert_production_ready(df)
    print("assert_production_ready: OK")
    sys.exit(0)

audit = df[df["seed_source"].astype(str).str.lower() != "winobias"].copy()
audit.to_parquet(path, index=False)
print("Removed {} WinoBias rows ({} -> {} audit rows)".format(n_wino, n_before, len(audit)))

df2 = pd.read_parquet(path)
assert_production_ready(df2)
print("assert_production_ready: OK")
print("Sources:", df2["seed_source"].value_counts().to_dict())
print("Rows:", len(df2), "Seeds:", df2["seed_id"].nunique())
