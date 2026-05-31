"""
File: Dataset/validate_pentad.py
Purpose: Schema and completeness validation for the pentad probe dataset.
         Fails loudly on any integrity violation.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the MIRAGE codebase. See README.md for full project context.
"""

import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEEDS_DIR

logger = logging.getLogger(__name__)

# Every seed must have exactly these (slot, subvariant) pairs.
# Slot c is verified by count (EXPECTED_C_COUNT) because subvariant names
# depend on the equivalence-set token chosen per seed.
REQUIRED_SLOT_SUBVARIANTS: list[tuple[str, str]] = [
    ("a", "surface"),
    ("b", "iso_control"),
    ("d", "d_valid"),
    ("d", "d_harmful"),
    ("e", "e1_neutral"),
    ("e", "e2_stereo_push"),
    ("e", "e3_antistereo_push"),
]

EXPECTED_C_COUNT = 5
TOTAL_EXPECTED = 12  # 1 + 1 + 5 + 2 + 3

REQUIRED_COLUMNS = {
    "seed_id",
    "seed_source",
    "seed_category",
    "prompt_id",
    "slot",
    "subvariant",
    "prompt_text",
}


def validate_schema(df: pd.DataFrame) -> None:
    """Check required columns are present and non-null."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Pentad dataset missing columns: {missing}")

    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' has {null_count} null values.")

    logger.info("Schema validation passed. Columns: %s", list(df.columns))


def validate_completeness(df: pd.DataFrame) -> list[str]:
    """
    Check every seed has all 12 prompt variants:
    - slot a: surface (1)
    - slot b: iso_control (1)
    - slot c: 5 counterfactual swaps (count check only -- token names vary)
    - slot d: d_valid, d_harmful (2)
    - slot e: e1_neutral, e2_stereo_push, e3_antistereo_push (3)
    Returns list of problem descriptions (empty if all OK).
    """
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        n_total = len(group)
        n_c = (group["slot"] == "c").sum()

        # Check slot-c count
        if n_c != EXPECTED_C_COUNT:
            problems.append(
                f"{seed_id}: expected {EXPECTED_C_COUNT} slot-c variants, got {n_c}"
            )

        # Check each required (slot, subvariant) pair exists
        for req_slot, req_sub in REQUIRED_SLOT_SUBVARIANTS:
            present = ((group["slot"] == req_slot) & (group["subvariant"] == req_sub)).any()
            if not present:
                problems.append(
                    f"{seed_id}: missing slot='{req_slot}' subvariant='{req_sub}'"
                )

        # Check total count
        if n_total != TOTAL_EXPECTED:
            problems.append(
                f"{seed_id}: expected {TOTAL_EXPECTED} total prompts, got {n_total}"
            )

    if problems:
        for p in problems[:20]:
            logger.error("Completeness error: %s", p)
        raise RuntimeError(
            f"Completeness check failed: {len(problems)} seeds with errors. "
            "First errors logged above."
        )

    logger.info("Completeness check passed. All seeds have %d prompts.", TOTAL_EXPECTED)
    return problems


def validate_no_empty_prompts(df: pd.DataFrame) -> None:
    """Fail if any prompt_text is empty or whitespace only."""
    empty = df[df["prompt_text"].str.strip() == ""]
    if len(empty) > 0:
        raise ValueError(
            f"{len(empty)} prompts have empty prompt_text. "
            f"Affected prompt_ids: {empty['prompt_id'].tolist()[:10]}"
        )
    logger.info("No empty prompts found.")


def validate_duplicate_prompt_ids(df: pd.DataFrame) -> None:
    """Fail if any prompt_id appears more than once."""
    dupes = df[df["prompt_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} rows have duplicate prompt_ids: "
            f"{dupes['prompt_id'].unique().tolist()[:10]}"
        )
    logger.info("No duplicate prompt_ids found.")


def run_all_validations(df: pd.DataFrame) -> None:
    """Run the full validation suite. Raises on any failure."""
    logger.info("Starting pentad validation on %d rows ...", len(df))
    validate_schema(df)
    validate_no_empty_prompts(df)
    validate_duplicate_prompt_ids(df)
    validate_completeness(df)
    logger.info("All pentad validations PASSED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error("Pentad dataset not found at %s. Run pentad_generator.py first.", pentad_path)
        sys.exit(1)
    df = pd.read_parquet(pentad_path)
    run_all_validations(df)
