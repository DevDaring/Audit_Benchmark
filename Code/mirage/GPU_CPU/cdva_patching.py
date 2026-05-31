"""
File: GPU_CPU/cdva_patching.py
Purpose: Causal Discriminative Validity Audit (CDVA) -- activation patching
         for all 4 OSM models across the counterfactual (c) probe variants.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
    -- CDVA discriminative validity test, FM2.
  - Meng et al. (2022). "Locating and Editing Factual Associations in GPT."
    NeurIPS 2022. https://arxiv.org/abs/2202.05262 -- activation patching.
  - Pearl (2009). Causality. Cambridge University Press.
    -- do-calculus / interventional framing.
  - Liu et al. (2026). "DIFFHEADS." AAAI 2026.

Part of the MIRAGE codebase. See README.md for full project context.
"""

import itertools
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, ensure_dirs
from GPU_CPU.utils_attention import _get_token_position, patch_activation

logger = logging.getLogger(__name__)

_CDVA_PATH = RESULTS_DIR / "cdva_results.parquet"

_MAX_DELTA_FALLBACK = 10.0  # Normalization upper bound if max_delta cannot be computed


def _cdva_pair_score(delta_logit: float, max_delta: float) -> float:
    """
    CDVA pairwise score: 1 - min(|delta_logit| / max_delta, 1.0)
    Higher = more causally invariant (better).
    """
    if max_delta <= 0:
        return 1.0
    return float(1.0 - min(abs(delta_logit) / max_delta, 1.0))


def run_cdva_for_seed(
    seed_id: str,
    c_variants: pd.DataFrame,
    model_cfg: dict,
    model: Any,
    tokenizer: Any,
    run_id: str,
) -> list[dict]:
    """
    Run CDVA for all C(5,2)=10 pairwise comparisons of slot-c variants
    for a single seed and model.

    Returns
    -------
    list[dict]
        One dict per pair.
    """
    rows: list[dict] = []
    model_name = model_cfg["name"]
    patching_lib = model_cfg["patching_lib"]
    try:
        model_version = model.config._name_or_path
    except Exception:
        model_version = model_cfg["hf_id"]

    variant_list = c_variants.to_dict("records")
    if len(variant_list) < 2:
        logger.warning("Seed %s has fewer than 2 slot-c variants; skipping CDVA.", seed_id)
        return rows

    delta_logits: list[float] = []
    pair_data: list[dict] = []

    for va, vb in itertools.combinations(variant_list, 2):
        prompt_a = str(va.get("prompt_text", ""))
        prompt_b = str(vb.get("prompt_text", ""))
        subvariant_a = str(va.get("subvariant", ""))
        subvariant_b = str(vb.get("subvariant", ""))
        swap_a = str(va.get("swap_token", ""))
        swap_b = str(vb.get("swap_token", ""))

        pos_a = _get_token_position(tokenizer, prompt_a, swap_a) if swap_a else None
        pos_b = _get_token_position(tokenizer, prompt_b, swap_b) if swap_b else None

        if pos_a is None or pos_b is None:
            pos_a = 1  # fallback to position 1 (after BOS)
            pos_b = 1

        # Bias answer: use a heuristic (first non-whitespace token of a generic bias answer)
        bias_answer = "Yes"  # Placeholder; ideally per-seed from dataset

        success_flag = True
        failure_reason = ""
        delta_logit = 0.0

        try:
            delta_logit = patch_activation(
                model, tokenizer,
                prompt_a, prompt_b,
                pos_a, pos_b,
                bias_answer,
                patching_lib,
            )
        except Exception as exc:
            logger.warning(
                "Patching failed for seed %s, pair (%s, %s): %s",
                seed_id, subvariant_a, subvariant_b, exc,
            )
            success_flag = False
            failure_reason = str(exc)

        delta_logits.append(delta_logit)
        pair_data.append(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed_id": seed_id,
                "model_name": model_name,
                "model_version": model_version,
                "pair_A_subvariant": subvariant_a,
                "pair_B_subvariant": subvariant_b,
                "delta_logit": delta_logit,
                "cdva_pair_score": 0.0,  # filled below
                "success_flag": success_flag,
                "failure_reason": failure_reason,
            }
        )

    # Normalise with max observed delta across this seed
    max_delta = max((abs(d) for d in delta_logits), default=_MAX_DELTA_FALLBACK)
    if max_delta == 0:
        max_delta = _MAX_DELTA_FALLBACK

    for i, pair in enumerate(pair_data):
        pair["cdva_pair_score"] = _cdva_pair_score(pair["delta_logit"], max_delta)

    return pair_data


def compute_cdva_seed_score(pair_rows: list[dict]) -> float:
    """Mean cdva_pair_score across all pairs for a seed."""
    scores = [r["cdva_pair_score"] for r in pair_rows if r["success_flag"]]
    return float(sum(scores) / len(scores)) if scores else 0.0


def run_cdva(
    pentad_df: pd.DataFrame,
    models: dict[str, tuple[Any, Any]],
    run_id: str,
) -> pd.DataFrame:
    """
    Run CDVA for all seeds and all OSM models.
    Writes incremental results to cdva_results.parquet.

    Returns
    -------
    pd.DataFrame
    """
    ensure_dirs()

    # Load existing results
    if _CDVA_PATH.exists():
        existing = pd.read_parquet(_CDVA_PATH)
        logger.info("Loaded %d existing CDVA results.", len(existing))
    else:
        existing = pd.DataFrame()

    completed: set[tuple] = set()
    if len(existing) > 0 and "success_flag" in existing.columns:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed.add((row["seed_id"], row["model_name"]))

    c_variants = pentad_df[pentad_df["slot"] == "c"].copy()
    seed_ids = c_variants["seed_id"].unique().tolist()

    all_rows: list[dict] = []
    if len(existing) > 0:
        all_rows.extend(existing.to_dict("records"))

    for model_cfg in OSM_MODELS:
        model_name = model_cfg["name"]
        if model_name not in models:
            logger.warning("Model '%s' not loaded, skipping CDVA.", model_name)
            continue

        model, tokenizer = models[model_name]
        logger.info("CDVA: model=%s, %d seeds ...", model_name, len(seed_ids))

        for i, seed_id in enumerate(seed_ids):
            if (seed_id, model_name) in completed:
                continue

            seed_c = c_variants[c_variants["seed_id"] == seed_id]
            try:
                pair_rows = run_cdva_for_seed(seed_id, seed_c, model_cfg, model, tokenizer, run_id)
                all_rows.extend(pair_rows)
            except Exception as exc:
                logger.error("CDVA failed for seed %s, model %s: %s", seed_id, model_name, exc)

            if (i + 1) % 25 == 0:
                df_partial = pd.DataFrame(all_rows)
                df_partial.to_parquet(_CDVA_PATH, index=False)
                logger.info("  CDVA checkpoint: %d seeds done.", i + 1)

    final = pd.DataFrame(all_rows)
    if len(final) > 0:
        final = (
            final.sort_values("timestamp_utc")
            .drop_duplicates(
                subset=["seed_id", "model_name", "pair_A_subvariant", "pair_B_subvariant"],
                keep="last",
            )
            .reset_index(drop=True)
        )
        final.to_parquet(_CDVA_PATH, index=False)

    logger.info("CDVA complete. Total pair rows: %d", len(final))
    return final
