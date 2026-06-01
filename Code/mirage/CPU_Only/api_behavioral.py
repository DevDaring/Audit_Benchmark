"""
File: CPU_Only/api_behavioral.py
Purpose: Behavioral evaluation across the 4 API models with retry/fallback
         policy specified in spec Section 4.2.

Temperature-variance pass (FM4):
  After the deterministic pass (sample_index=0, temperature=0.0), a second
  pass runs slot-a 5 times at temperature=0.7 (sample_index 1-5).  This
  provides data for FM4 (criterion leakage / answer variance under sampling).
  Previously this pass was absent for API models (review finding B5).

gold_answer is copied from pentad_df into every behavioral result row so
that scoring.py can compare parsed_answer to gold_answer without a join.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
  - Nadeem et al. (2021). StereoSet. ACL-IJCNLP 2021.

Part of the MIRAGE codebase. See README.md for full project context.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_MODELS, RESULTS_DIR, RESEARCH_SYSTEM_PROMPT, ensure_dirs
from parse_utils import parse_model_response, repair_json
from results_utils import dedup_behavioral, reparse_failed_rows

logger = logging.getLogger(__name__)

_BEHAVIORAL_PATH = RESULTS_DIR / "behavioral_results.parquet"

_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT

# Number of stochastic samples for the FM4 variance pass (slot-a only)
_FM4_N_SAMPLES = 5
_FM4_TEMPERATURE = 0.7


def _parse_response(raw: str, prompt_id: str) -> tuple[dict | None, str]:
    """Parse raw response; try shared parser, then judge fallback."""
    if not raw:
        return None, "failed"

    success, answer, conf, rationale, method, reason = parse_model_response(raw)
    if success:
        return {"answer": answer, "confidence": conf, "rationale": rationale}, method

    repaired = repair_json(raw)
    if repaired is not None and str(repaired.get("answer", "")).strip():
        return repaired, "json"

    from CPU_Only.judge_router import judge
    parsed, method = judge(raw, provider="gemini")
    return parsed, method


def _build_messages(prompt_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def _call_api_model(
    model_cfg: dict,
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Route to the appropriate client based on primary_route."""
    primary = model_cfg["primary_route"]
    model_id = model_cfg["model_id"]

    if primary == "bedrock":
        from CPU_Only.api_clients.bedrock_client import call_bedrock_with_fallback
        return call_bedrock_with_fallback(model_id, messages, max_tokens, temperature=temperature)
    elif primary == "gcp":
        from CPU_Only.api_clients.gemini_client import call_gemini_with_roundrobin
        return call_gemini_with_roundrobin(messages, max_tokens, temperature=temperature)
    elif primary == "mistral":
        from CPU_Only.api_clients.mistral_client import call_mistral_with_roundrobin
        return call_mistral_with_roundrobin(messages, max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Unknown primary_route: '{primary}'")


def _evaluate_single_prompt(
    model_cfg: dict,
    prompt_id: str,
    prompt_text: str,
    gold_answer: str,
    run_id: str,
    seed_id: str,
    seed_source: str,
    seed_category: str,
    seed_subcategory: str,
    slot: str,
    subvariant: str,
    max_tokens: int,
    sample_index: int,
    temperature: float,
) -> dict:
    """Run a single API call and return a result row dict."""
    messages = _build_messages(prompt_text)
    t0 = time.monotonic()
    api_result = _call_api_model(model_cfg, messages, max_tokens, temperature=temperature)
    latency_ms = int((time.monotonic() - t0) * 1000)

    model_name = model_cfg["name"]
    raw = api_result.get("raw_response", "")
    parsed = None
    parse_method = "failed"
    parsed_answer = ""
    parsed_confidence = 0.0
    parsed_rationale = ""
    success_flag = api_result.get("success_flag", False)
    failure_reason = api_result.get("failure_reason", "")

    if success_flag and raw:
        parsed, parse_method = _parse_response(raw, prompt_id)
        if parsed:
            parsed_answer = str(parsed.get("answer", ""))
            parsed_confidence = float(parsed.get("confidence", 0.0))
            parsed_rationale = str(parsed.get("rationale", ""))
        else:
            parse_method = "failed"
            success_flag = False
            failure_reason = "parse_error"

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed_id": seed_id,
        "seed_source": seed_source,
        "seed_category": seed_category,
        "seed_subcategory": seed_subcategory,
        "prompt_id": prompt_id,
        "slot": slot,
        "subvariant": subvariant,
        "model_name": model_name,
        "model_provider": model_cfg.get("primary_route", ""),
        "model_version": model_cfg.get("model_id", ""),
        "route_used": api_result.get("route_used", ""),
        "key_index": api_result.get("key_index", -1),
        "attempt_count": api_result.get("attempt_count", 1),
        "prompt_text": prompt_text,
        "gold_answer": gold_answer,
        "raw_response": raw,
        "parsed_answer": parsed_answer,
        "parsed_confidence": parsed_confidence,
        "parsed_rationale": parsed_rationale,
        "parse_method": parse_method,
        "success_flag": success_flag,
        "failure_reason": failure_reason,
        "latency_ms": api_result.get("latency_ms", latency_ms),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "sample_index": sample_index,
    }


def evaluate_api_model(
    model_cfg: dict,
    pentad_df: pd.DataFrame,
    run_id: str,
    completed_keys: set[tuple],
    max_tokens: int = 256,
    sample_index: int = 0,
    temperature: float = 0.0,
) -> pd.DataFrame:
    """
    Evaluate a single API model on the pentad dataset.
    Skips rows already present in completed_keys.

    Parameters
    ----------
    sample_index : int
        0 = deterministic pass (temperature=0.0).
        1-5 = stochastic FM4 pass (temperature=0.7, slot-a only).
    temperature : float
        Sampling temperature.  0.0 for deterministic, 0.7 for FM4 pass.
    """
    model_name = model_cfg["name"]
    rows: list[dict] = []
    total = len(pentad_df)

    for i, (_, prow) in enumerate(pentad_df.iterrows()):
        prompt_id = prow["prompt_id"]
        slot = prow.get("slot", "")
        subvariant = prow.get("subvariant", "")

        # FM4 variance pass only runs on slot-a
        if sample_index > 0 and slot != "a":
            continue

        if (prompt_id, model_name, sample_index) in completed_keys:
            continue

        prompt_text = str(prow.get("prompt_text", ""))
        if not prompt_text.strip() or prompt_text.strip().lower() == "none":
            logger.debug(
                "Skipping invalid prompt_text for prompt_id=%s (value=%r)",
                prompt_id, prompt_text[:80],
            )
            continue

        gold_answer = str(prow.get("gold_answer", ""))

        row = _evaluate_single_prompt(
            model_cfg=model_cfg,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            gold_answer=gold_answer,
            run_id=run_id,
            seed_id=str(prow.get("seed_id", "")),
            seed_source=str(prow.get("seed_source", "")),
            seed_category=str(prow.get("seed_category", "")),
            seed_subcategory=str(prow.get("seed_subcategory", "")),
            slot=slot,
            subvariant=subvariant,
            max_tokens=max_tokens,
            sample_index=sample_index,
            temperature=temperature,
        )
        rows.append(row)

        if (i + 1) % 50 == 0:
            logger.info(
                "API %s (sample_index=%d): %d/%d prompts done.",
                model_name, sample_index, i + 1, total,
            )

    return pd.DataFrame(rows)


def run_api_behavioral(
    pentad_df: pd.DataFrame,
    run_id: str,
    max_tokens: int = 256,
) -> pd.DataFrame:
    """
    Run behavioral evaluation for all 4 API models with resume logic.

    Two passes per model:
      Pass 1 (sample_index=0, temp=0.0): all slots.
      Pass 2 (sample_index=1..5, temp=0.7): slot-a only (FM4 variance).

    Appends to existing behavioral_results.parquet.
    """
    ensure_dirs()

    if _BEHAVIORAL_PATH.exists():
        existing = dedup_behavioral(reparse_failed_rows(pd.read_parquet(_BEHAVIORAL_PATH)))
        if len(existing) > 0:
            existing.to_parquet(_BEHAVIORAL_PATH, index=False)
        logger.info("Loaded %d existing results for resume (deduped).", len(existing))
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple] = set()
    if len(existing) > 0 and "success_flag" in existing.columns:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))

    working = existing

    for model_cfg in API_MODELS:
        model_name = model_cfg["name"]
        logger.info("API evaluation: model=%s ...", model_name)

        # Pass 1: deterministic (all slots, sample_index=0, temp=0.0)
        result_df = evaluate_api_model(
            model_cfg, pentad_df, run_id, completed_keys,
            max_tokens=max_tokens, sample_index=0, temperature=0.0,
        )
        if len(result_df) > 0:
            working = pd.concat([working, result_df], ignore_index=True)
            working = dedup_behavioral(working)
            working.to_parquet(_BEHAVIORAL_PATH, index=False)
            for _, row in result_df[result_df["success_flag"] == True].iterrows():  # noqa: E712
                completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))
            logger.info("  Saved %d new rows for %s (deterministic pass).", len(result_df), model_name)

        # Pass 2: stochastic FM4 variance (slot-a only, 5 samples at temp=0.7)
        logger.info("  FM4 variance pass: model=%s, %d samples at temp=%.1f ...",
                    model_name, _FM4_N_SAMPLES, _FM4_TEMPERATURE)
        for sample_idx in range(1, _FM4_N_SAMPLES + 1):
            var_df = evaluate_api_model(
                model_cfg, pentad_df, run_id, completed_keys,
                max_tokens=max_tokens,
                sample_index=sample_idx,
                temperature=_FM4_TEMPERATURE,
            )
            if len(var_df) > 0:
                working = pd.concat([working, var_df], ignore_index=True)
                working = dedup_behavioral(working)
                working.to_parquet(_BEHAVIORAL_PATH, index=False)
                for _, row in var_df[var_df["success_flag"] == True].iterrows():  # noqa: E712
                    completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))
                logger.info(
                    "    Saved %d rows for %s (sample_index=%d).",
                    len(var_df), model_name, sample_idx,
                )

    final = dedup_behavioral(working) if len(working) > 0 else pd.DataFrame()
    if len(final) > 0:
        final.to_parquet(_BEHAVIORAL_PATH, index=False)

    logger.info("API behavioral evaluation complete. Total rows: %d", len(final))
    return final
