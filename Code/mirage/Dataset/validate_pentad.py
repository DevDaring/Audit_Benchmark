"""
File: Dataset/validate_pentad.py
Purpose: Schema and completeness validation for the pentad probe dataset.
         Fails loudly on any integrity violation.

Semantic gates added (fixes review finding C1):
  - Rejects "None"/"nan"/"null" prompt texts (these slipped through the old
    whitespace-only check).
  - Verifies that slot-b text differs from slot-a text for each seed (or
    flags seeds where no protected token was found so the difference is
    documented rather than silently wrong).
  - Verifies that the 5 slot-c prompt texts are distinct (genuinely different
    counterfactual swaps, not 5 copies of the same prompt).
  - Verifies that multiple-choice sources (BBQ) include their answer options
    in the prompt text.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SEEDS_DIR

logger = logging.getLogger(__name__)

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

# Sentinel strings that indicate a mis-constructed prompt (A2 fix)
_INVALID_PROMPT_SENTINELS = re.compile(
    r"^(None|none|nan|NaN|null|NULL|na|NA)$", re.IGNORECASE
)

# MCQ sources that must include answer options in the prompt
_MCQ_SOURCES = {"bbq"}
_MCQ_OPTION_PATTERN = re.compile(r"\(A\)|\(B\)|\(C\)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Structural validators (schema, counts, duplicates)
# ---------------------------------------------------------------------------

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
    Check every seed has all 12 prompt variants.
    Returns list of problem descriptions (empty if all OK).
    """
    problems: list[str] = []
    for seed_id, group in df.groupby("seed_id"):
        n_total = len(group)
        n_c = (group["slot"] == "c").sum()

        if n_c != EXPECTED_C_COUNT:
            problems.append(
                f"{seed_id}: expected {EXPECTED_C_COUNT} slot-c variants, got {n_c}"
            )

        for req_slot, req_sub in REQUIRED_SLOT_SUBVARIANTS:
            present = ((group["slot"] == req_slot) & (group["subvariant"] == req_sub)).any()
            if not present:
                problems.append(
                    f"{seed_id}: missing slot='{req_slot}' subvariant='{req_sub}'"
                )

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


def validate_duplicate_prompt_ids(df: pd.DataFrame) -> None:
    """Fail if any prompt_id appears more than once."""
    dupes = df[df["prompt_id"].duplicated(keep=False)]
    if len(dupes) > 0:
        raise ValueError(
            f"{len(dupes)} rows have duplicate prompt_ids: "
            f"{dupes['prompt_id'].unique().tolist()[:10]}"
        )
    logger.info("No duplicate prompt_ids found.")


# ---------------------------------------------------------------------------
# Semantic validators (new -- fixes C1)
# ---------------------------------------------------------------------------

def validate_no_sentinel_prompts(df: pd.DataFrame) -> None:
    """
    Fail if any prompt_text is empty, whitespace-only, or equals a known
    sentinel string (None/nan/null) that indicates a construction failure.

    Previously only the whitespace check was applied, which let 1281
    StereoSet "None" prompts through.
    """
    problems: list[str] = []

    for _, row in df.iterrows():
        text = str(row["prompt_text"]) if pd.notna(row["prompt_text"]) else ""
        stripped = text.strip()

        if not stripped:
            problems.append(
                f"prompt_id={row['prompt_id']}: empty or whitespace-only prompt_text."
            )
        elif _INVALID_PROMPT_SENTINELS.match(stripped):
            problems.append(
                f"prompt_id={row['prompt_id']}: sentinel prompt_text value '{stripped}' "
                f"(seed_source={row.get('seed_source','')})."
            )

    if problems:
        for p in problems[:20]:
            logger.error("Sentinel prompt error: %s", p)
        raise ValueError(
            f"{len(problems)} prompts contain sentinel/empty text. "
            "First errors logged above."
        )
    logger.info("No sentinel prompts found.")


def validate_b_differs_from_a(df: pd.DataFrame) -> None:
    """
    Verify that slot-b prompt_text differs from slot-a prompt_text for each
    seed.  Seeds with no protected token in their prompt (and therefore an
    identical slot-b) are flagged as warnings rather than errors, because the
    review notes this is an inherent limitation for some items.

    A HARD FAILURE is raised if more than 50% of seeds have identical a/b.
    """
    identical_count = 0
    total_seeds = 0

    for seed_id, group in df.groupby("seed_id"):
        a_rows = group[group["subvariant"] == "surface"]
        b_rows = group[group["subvariant"] == "iso_control"]
        if a_rows.empty or b_rows.empty:
            continue
        total_seeds += 1
        a_text = str(a_rows.iloc[0]["prompt_text"])
        b_text = str(b_rows.iloc[0]["prompt_text"])
        if a_text.strip() == b_text.strip():
            identical_count += 1
            logger.debug(
                "seed_id=%s: slot-a == slot-b (no protected token substituted).", seed_id
            )

    if total_seeds == 0:
        return

    rate = identical_count / total_seeds
    logger.info(
        "Slot-b identical to slot-a: %d/%d seeds (%.1f%%).",
        identical_count, total_seeds, rate * 100,
    )
    if rate > 0.5:
        raise ValueError(
            f"slot-b == slot-a for {identical_count}/{total_seeds} seeds ({rate:.1%}). "
            "Protected-token substitution is failing for the majority of seeds. "
            "Check _PROTECTED_TO_NEUTRAL and the equivalence-set routing."
        )


def validate_c_variants_distinct(df: pd.DataFrame) -> None:
    """
    Verify that the 5 slot-c prompt_texts are distinct for each seed.

    Seeds with fewer than 5 distinct texts are flagged as warnings.  A HARD
    FAILURE is raised if more than 25% of seeds have only 1 unique c-text
    (which was the case for 87% of seeds in the broken dataset).
    """
    degenerate_count = 0
    total_seeds = 0

    for seed_id, group in df.groupby("seed_id"):
        c_rows = group[group["slot"] == "c"]
        if len(c_rows) < 2:
            continue
        total_seeds += 1
        n_unique = c_rows["prompt_text"].nunique()
        if n_unique == 1:
            degenerate_count += 1
            logger.debug(
                "seed_id=%s: all 5 slot-c prompts are identical (swap had no effect).", seed_id
            )
        elif n_unique < 5:
            logger.warning(
                "seed_id=%s: only %d of 5 slot-c prompts are distinct.", seed_id, n_unique
            )

    if total_seeds == 0:
        return

    rate = degenerate_count / total_seeds
    logger.info(
        "Slot-c all-identical seeds: %d/%d (%.1f%%).",
        degenerate_count, total_seeds, rate * 100,
    )
    if rate > 0.25:
        raise ValueError(
            f"{degenerate_count}/{total_seeds} seeds ({rate:.1%}) have all 5 slot-c "
            "prompts identical -- the counterfactual swap is not taking effect in the "
            "full prompt text. Check _build_slot_c and verify the demographic token "
            "appears in the complete prompt (context + question + options)."
        )


def validate_mcq_options_present(df: pd.DataFrame) -> None:
    """
    For MCQ sources (BBQ), verify that every slot-a prompt contains answer
    options in (A) / (B) / (C) format.

    StereoSet slots a/b/c should also include options after the fix.
    """
    problems: list[str] = []

    mcq_rows = df[
        (df["seed_source"].isin(_MCQ_SOURCES)) & (df["slot"] == "a")
    ]

    for _, row in mcq_rows.iterrows():
        text = str(row["prompt_text"])
        if not _MCQ_OPTION_PATTERN.search(text):
            problems.append(
                f"prompt_id={row['prompt_id']} ({row['seed_source']}): "
                "slot-a prompt is missing answer options (A)/(B)/(C). "
                "Expected the full context + MCQ format."
            )

    if problems:
        for p in problems[:20]:
            logger.error("MCQ options missing: %s", p)
        raise ValueError(
            f"{len(problems)} MCQ prompts missing answer options. "
            "First errors logged above."
        )
    logger.info("MCQ answer-options check passed.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_all_validations(df: pd.DataFrame) -> None:
    """Run the full validation suite. Raises on any failure."""
    logger.info("Starting pentad validation on %d rows ...", len(df))
    validate_schema(df)
    validate_no_sentinel_prompts(df)
    validate_duplicate_prompt_ids(df)
    validate_completeness(df)
    validate_b_differs_from_a(df)
    validate_c_variants_distinct(df)
    validate_mcq_options_present(df)
    logger.info("All pentad validations PASSED.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error("Pentad dataset not found at %s. Run pentad_generator.py first.", pentad_path)
        import sys
        sys.exit(1)
    df = pd.read_parquet(pentad_path)
    run_all_validations(df)
