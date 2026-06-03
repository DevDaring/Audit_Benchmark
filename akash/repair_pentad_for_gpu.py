"""
Repair pentad for production GPU run:
  - Remove WinoBias (held out)
  - Keep existing slot d/e API rows
  - Regenerate slot a/b/c from seeds (fixes None prompt_text)
  - Validate and write manifest
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/koushikdeb2009/Audit_Benchmark/Code/mirage")
from config import RANDOM_SEED, SEEDS_DIR
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_PENTAD_PATH = SEEDS_DIR / "pentad_dataset.parquet"
_AUDIT_SOURCES = frozenset({"bbq", "crows_pairs", "stereoset"})


def main() -> bool:
    setup_logging()
    if not _PENTAD_PATH.exists():
        logger.error("Missing %s", _PENTAD_PATH)
        return False

    old = pd.read_parquet(_PENTAD_PATH)
    old = old[old["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)]
    logger.info("Audit rows in existing pentad: %d", len(old))

    api_df = old[old["slot"].isin(["d", "e"])].copy()
    logger.info("Keeping %d API rows (d/e)", len(api_df))

    from Dataset.sample_seeds import sample_seeds
    from Dataset.pentad_generator import generate_pentad_deterministic

    main_seeds, _ = sample_seeds()
    main_seeds = main_seeds[
        main_seeds["seed_source"].astype(str).str.lower().isin(_AUDIT_SOURCES)
    ].reset_index(drop=True)
    logger.info("Regenerating a/b/c for %d audit seeds", len(main_seeds))

    rng = np.random.default_rng(seed=RANDOM_SEED)
    det_rows = generate_pentad_deterministic(main_seeds, rng)
    det_df = pd.DataFrame(det_rows)
    ok_ids = set(det_df["seed_id"].unique())
    logger.info("Generated %d det rows for %d seeds", len(det_df), len(ok_ids))

    # Keep API rows only for seeds that have valid det rows
    api_df = api_df[api_df["seed_id"].isin(ok_ids)].copy()
    logger.info("API rows after seed filter: %d", len(api_df))

    combined = pd.concat([det_df, api_df], ignore_index=True)
    combined = combined.sort_values(["seed_id", "slot", "subvariant"]).reset_index(drop=True)

    # Propagate gold_answer to API rows from det slot-a
    if "gold_answer" in det_df.columns:
        gold_by_seed = (
            det_df[det_df["slot"] == "a"][["seed_id", "gold_answer"]]
            .drop_duplicates("seed_id")
            .set_index("seed_id")["gold_answer"]
        )
        if "gold_answer" not in combined.columns:
            combined["gold_answer"] = combined["seed_id"].map(gold_by_seed)
        else:
            combined["gold_answer"] = combined["gold_answer"].fillna(
                combined["seed_id"].map(gold_by_seed)
            )

    combined.to_parquet(_PENTAD_PATH, index=False)
    logger.info("Saved %d rows -> %s", len(combined), _PENTAD_PATH)

    from Dataset.validate_pentad import assert_production_ready, write_pentad_manifest

    assert_production_ready(combined)
    manifest = write_pentad_manifest(combined)
    logger.info("Production ready. Manifest: %s", json.dumps(manifest, indent=2))
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ok = main()
    sys.exit(0 if ok else 1)
