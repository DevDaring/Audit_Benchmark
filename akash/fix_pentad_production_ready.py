"""
Repair pentad_dataset.parquet for production GPU pipeline:
  1. Remove WinoBias rows (held-out benchmark).
  2. Add gold_answer column (required by validate_pentad / scoring).
  3. Re-validate and write manifest.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/koushikdeb2009/Audit_Benchmark/Code/mirage")

import pandas as pd
from config import SEEDS_DIR
from Dataset.pentad_generator import _build_full_prompt
from Dataset.validate_pentad import assert_production_ready, write_pentad_manifest

path = SEEDS_DIR / "pentad_dataset.parquet"
seeds_path = SEEDS_DIR / "seeds.parquet"

df = pd.read_parquet(path)
seeds = pd.read_parquet(seeds_path)

print("Before: {} rows, sources={}".format(
    len(df), df["seed_source"].value_counts().to_dict()
))

# 1. Strip WinoBias
wino_mask = df["seed_source"].astype(str).str.lower() == "winobias"
if wino_mask.any():
    df = df[~wino_mask].reset_index(drop=True)
    print("Removed {} WinoBias rows".format(int(wino_mask.sum())))

# 2. Add gold_answer per seed_id
if "gold_answer" not in df.columns or df["gold_answer"].isnull().any():
    gold_map: dict[str, str] = {}
    missing_seeds: list[str] = []
    for seed_id in df["seed_id"].unique():
        match = seeds[seeds["seed_id"] == seed_id]
        if match.empty:
            missing_seeds.append(str(seed_id))
            continue
        _, gold = _build_full_prompt(match.iloc[0].to_dict())
        gold_map[str(seed_id)] = str(gold)

    if missing_seeds:
        raise RuntimeError(
            "No seeds.parquet row for {} seed_ids (first: {})".format(
                len(missing_seeds), missing_seeds[:3]
            )
        )

    df["gold_answer"] = df["seed_id"].astype(str).map(gold_map)
    nulls = int(df["gold_answer"].isnull().sum())
    if nulls:
        raise RuntimeError("{} rows still missing gold_answer".format(nulls))
    print("Added gold_answer for {} seeds".format(len(gold_map)))

df.to_parquet(path, index=False)
print("Saved:", path)

df2 = pd.read_parquet(path)
assert_production_ready(df2)
manifest = write_pentad_manifest(df2)
print("assert_production_ready: OK")
print("Manifest:", manifest)
print("After: {} rows, {} seeds".format(len(df2), df2["seed_id"].nunique()))
