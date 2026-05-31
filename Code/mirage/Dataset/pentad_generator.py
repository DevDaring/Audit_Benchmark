"""
File: Dataset/pentad_generator.py
Purpose: Orchestrates generation of all 12 probe variants (5 slots) per seed.
         Slots (a), (b), (c) are deterministic; (d) and (e) call DeepSeek API.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
    -- pentad probe design, Epistematics framework
  - Kusner et al. (2017). "Counterfactual Fairness." NeurIPS 2017.
    -- counterfactual swap rationale for slot (c)
  - Shaikh et al. (2023). "On Second Thought, Let's Not Think Step by Step!"
    ACL 2023. -- CoT attack rationale for slot (e)

Part of the MIRAGE codebase. See README.md for full project context.
"""

import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED, SEEDS_DIR, ensure_dirs

logger = logging.getLogger(__name__)

_EQUIV_SETS_PATH = Path(__file__).resolve().parent / "equivalence_sets.yaml"
_PENTAD_PATH = SEEDS_DIR / "pentad_dataset.parquet"

# Mapping from neutral group tokens to protected group category
_PROTECTED_TO_NEUTRAL: dict[str, str] = {
    # gender
    "man": "person",
    "woman": "person",
    "non-binary person": "person",
    "transgender man": "person",
    "transgender woman": "person",
    # race
    "White": "person",
    "Black": "person",
    "Asian": "person",
    "Hispanic": "person",
    "Latino": "person",
    # religion
    "Sikh": "person",
    "Hindu": "person",
    "Muslim": "person",
    "Buddhist": "person",
    "Christian": "person",
    "Jewish": "person",
}


def _load_equiv_sets() -> dict[str, list[str]]:
    with open(_EQUIV_SETS_PATH) as fh:
        return yaml.safe_load(fh)


def _find_category_for_token(token: str, equiv_sets: dict[str, list[str]]) -> str | None:
    for cat, tokens in equiv_sets.items():
        if token in tokens:
            return cat
    return None


def _build_slot_a(seed_row: dict) -> dict:
    """Slot (a): verbatim copy of the seed prompt."""
    text = seed_row.get("question") or seed_row.get("sent_more") or seed_row.get("sentence", "")
    return {
        "slot": "a",
        "subvariant": "surface",
        "prompt_text": str(text),
    }


def _build_slot_b(seed_row: dict, protected_to_neutral: dict[str, str]) -> dict:
    """Slot (b): isomorphic control -- replace protected-group token with neutral."""
    text = seed_row.get("question") or seed_row.get("sent_more") or seed_row.get("sentence", "")
    text = str(text)
    for token, neutral in protected_to_neutral.items():
        if token.lower() in text.lower():
            text = text.replace(token, neutral)
            break
    return {
        "slot": "b",
        "subvariant": "iso_control",
        "prompt_text": text,
    }


def _build_slot_c(
    seed_row: dict,
    equiv_sets: dict[str, list[str]],
    rng: np.random.Generator,
) -> list[dict]:
    """Slot (c): 5 counterfactual swaps through closed equivalence set.

    Category matching is bidirectional: cat_key matches seed_category if either
    is a substring of the other (case-insensitive), or if any equivalence token
    appears in the prompt text.  This ensures e.g. seed_category='Age' matches
    yaml key 'age_group', and seed_category='Race_x_gender' matches 'gender' via
    the text-token scan.
    """
    text = seed_row.get("question") or seed_row.get("sent_more") or seed_row.get("sentence", "")
    text = str(text)
    category = seed_row.get("seed_category", "")
    cat_lower = category.lower()

    # Find the best-matching equivalence set (bidirectional substring match + token scan)
    eq_tokens: list[str] = []
    for cat_key, tokens in equiv_sets.items():
        key_lower = cat_key.lower()
        if (
            key_lower in cat_lower
            or cat_lower in key_lower
            or any(t.lower() in text.lower() for t in tokens)
        ):
            eq_tokens = tokens
            break

    if not eq_tokens:
        # Fallback: generic person-neutral set (only when no category match exists)
        eq_tokens = ["person", "individual", "someone", "they", "one"]

    # Identify the original group token present in the text (if any)
    original_token = next(
        (t for t in eq_tokens if t.lower() in text.lower()), eq_tokens[0]
    )

    # Build candidate replacement tokens (exclude original to avoid identity swap)
    # replace=False ensures no within-sample duplicates
    other_tokens = [t for t in eq_tokens if t != original_token]
    n_pick = min(4, len(other_tokens))
    chosen = list(rng.choice(other_tokens, size=n_pick, replace=False)) if n_pick > 0 else []

    # variants: original + 4 others, deduplicated, capped at 5
    seen: set[str] = set()
    variants: list[str] = []
    for v in [original_token] + chosen:
        if v not in seen:
            seen.add(v)
            variants.append(v)
    variants = variants[:5]

    # Pad to exactly 5 if we ran short (only happens when equiv set has < 5 members)
    if len(variants) < 5:
        for t in eq_tokens:
            if t not in seen:
                seen.add(t)
                variants.append(t)
            if len(variants) == 5:
                break

    slots: list[dict] = []
    for variant_token in variants:
        swapped = text
        for token in eq_tokens:
            if token.lower() in swapped.lower():
                swapped = swapped.replace(token, variant_token)
                break
        # subvariant is the token name only (no redundant 'c_' prefix);
        # prompt_id becomes {seed_id}_c_{token} which is clean and unique.
        subvariant = variant_token.lower().replace(" ", "_").replace("-", "_")
        slots.append(
            {
                "slot": "c",
                "subvariant": subvariant,
                "prompt_text": swapped,
                "swap_token": variant_token,
            }
        )
    return slots


def _row_to_dict(row: Any) -> dict:
    if isinstance(row, dict):
        return row
    return row.to_dict()


def generate_pentad_deterministic(
    seeds_df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Generate slots (a), (b), (c) deterministically for every seed.
    Returns a list of row dicts to be combined into the pentad dataset.
    """
    equiv_sets = _load_equiv_sets()
    rows: list[dict] = []

    for _, seed_row in seeds_df.iterrows():
        seed_dict = _row_to_dict(seed_row)
        seed_id = seed_dict.get("seed_id", str(uuid.uuid4()))

        slot_a = _build_slot_a(seed_dict)
        slot_b = _build_slot_b(seed_dict, _PROTECTED_TO_NEUTRAL)
        slot_c_list = _build_slot_c(seed_dict, equiv_sets, rng)

        for variant in [slot_a, slot_b] + slot_c_list:
            prompt_id = f"{seed_id}_{variant['slot']}_{variant['subvariant']}"
            rows.append(
                {
                    "seed_id": seed_id,
                    "seed_source": seed_dict.get("seed_source", ""),
                    "seed_category": seed_dict.get("seed_category", ""),
                    "seed_subcategory": seed_dict.get("seed_subcategory", ""),
                    "prompt_id": prompt_id,
                    **variant,
                    "generated_by": "deterministic",
                    "generator_model": "",
                    "generator_timestamp": "",
                }
            )

    return rows


def build_pentad_dataset(
    seeds_df: pd.DataFrame,
    include_api_slots: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Orchestrate full pentad generation for all seeds.
    Slots (d) and (e) require DeepSeek API calls; set include_api_slots=False
    to skip them during dry runs or testing.

    Parameters
    ----------
    seeds_df : pd.DataFrame
    include_api_slots : bool
        If False, generate only (a)-(c) deterministically.
    force : bool
        Regenerate even if cached file exists.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()

    if _PENTAD_PATH.exists() and not force:
        logger.info("Pentad dataset cache hit: %s", _PENTAD_PATH)
        return pd.read_parquet(_PENTAD_PATH)

    rng = np.random.default_rng(seed=RANDOM_SEED)

    logger.info("Generating deterministic pentad slots (a, b, c) for %d seeds ...", len(seeds_df))
    rows = generate_pentad_deterministic(seeds_df, rng)
    logger.info("  Generated %d rows for slots a/b/c.", len(rows))

    if include_api_slots:
        from Dataset.context_shift_drafter import draft_context_shifts
        from Dataset.cot_attack_generator import generate_cot_attacks

        logger.info("Generating slot (d) -- context shift -- via DeepSeek API ...")
        d_rows = draft_context_shifts(seeds_df)
        rows.extend(d_rows)

        logger.info("Generating slot (e) -- CoT attack -- via DeepSeek API ...")
        e_rows = generate_cot_attacks(seeds_df)
        rows.extend(e_rows)

    df = pd.DataFrame(rows)
    df.to_parquet(_PENTAD_PATH, index=False)
    logger.info("Pentad dataset saved: %d rows -> %s", len(df), _PENTAD_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from Dataset.sample_seeds import sample_seeds
    main_seeds, _ = sample_seeds()
    pentad = build_pentad_dataset(main_seeds, include_api_slots=False)
    logger.info("Pentad (det-only): %d rows", len(pentad))
