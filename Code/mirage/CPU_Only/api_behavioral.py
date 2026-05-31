"""
File: CPU_Only/api_behavioral.py
Purpose: Behavioral evaluation across the 4 API models with retry/fallback
         policy specified in spec Section 4.2.

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

logger = logging.getLogger(__name__)

_BEHAVIORAL_PATH = RESULTS_DIR / "behavioral_results.parquet"

_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT


def _repair_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("{") and not raw.endswith("}"):
        raw = raw + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _build_messages(prompt_text: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def _call_api_model(model_cfg: dict, messages: list[dict], max_tokens: int) -> dict[str, Any]:
    """Route to the appropriate client based on primary_route."""
    primary = model_cfg["primary_route"]
    model_id = model_cfg["model_id"]

    if primary == "bedrock":
        from CPU_Only.api_clients.bedrock_client import call_bedrock_with_fallback
        return call_bedrock_with_fallback(model_id, messages, max_tokens)
    elif primary == "gcp":
        from CPU_Only.api_clients.gemini_client import call_gemini_with_roundrobin
        return call_gemini_with_roundrobin(messages, max_tokens)
    elif primary == "mistral":
        from CPU_Only.api_clients.mistral_client import call_mistral_with_roundrobin
        return call_mistral_with_roundrobin(messages, max_tokens)
    else:
        raise ValueError(f"Unknown primary_route: '{primary}'")


def _parse_response(raw: str, prompt_id: str) -> tuple[dict | None, str]:
    """Parse raw response; try deterministic repair; route to judge if needed."""
    if not raw:
        return None, "failed"

    # Direct parse
    try:
        return json.loads(raw), "json"
    except json.JSONDecodeError:
        pass

    # Repair attempt
    repaired = _repair_json(raw)
    if repaired is not None:
        return repaired, "json"

    # Judge (default: gemini)
    from CPU_Only.judge_router import judge
    parsed, method = judge(raw, provider="gemini")
    return parsed, method


def evaluate_api_model(
    model_cfg: dict,
    pentad_df: pd.DataFrame,
    run_id: str,
    completed_keys: set[tuple],
    max_tokens: int = 256,
    sample_index: int = 0,
) -> pd.DataFrame:
    """
    Evaluate a single API model on the pentad dataset.
    Skips rows already present in completed_keys.
    """
    model_name = model_cfg["name"]
    rows: list[dict] = []
    total = len(pentad_df)

    for i, (_, prow) in enumerate(pentad_df.iterrows()):
        prompt_id = prow["prompt_id"]
        if (prompt_id, model_name, sample_index) in completed_keys:
            continue

        prompt_text = str(prow.get("prompt_text", ""))
        if not prompt_text.strip():
            continue

        messages = _build_messages(prompt_text)
        t0 = time.monotonic()
        api_result = _call_api_model(model_cfg, messages, max_tokens)
        latency_ms = int((time.monotonic() - t0) * 1000)

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

        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed_id": prow.get("seed_id", ""),
                "seed_source": prow.get("seed_source", ""),
                "seed_category": prow.get("seed_category", ""),
                "seed_subcategory": prow.get("seed_subcategory", ""),
                "prompt_id": prompt_id,
                "slot": prow.get("slot", ""),
                "subvariant": prow.get("subvariant", ""),
                "model_name": model_name,
                "model_provider": model_cfg.get("primary_route", ""),
                "model_version": model_cfg.get("model_id", ""),
                "route_used": api_result.get("route_used", ""),
                "key_index": api_result.get("key_index", -1),
                "attempt_count": api_result.get("attempt_count", 1),
                "prompt_text": prompt_text,
                "raw_response": raw,
                "parsed_answer": parsed_answer,
                "parsed_confidence": parsed_confidence,
                "parsed_rationale": parsed_rationale,
                "parse_method": parse_method,
                "success_flag": success_flag,
                "failure_reason": failure_reason,
                "latency_ms": api_result.get("latency_ms", latency_ms),
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "sample_index": sample_index,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info("API %s: %d/%d prompts done.", model_name, i + 1, total)

    return pd.DataFrame(rows)


def run_api_behavioral(
    pentad_df: pd.DataFrame,
    run_id: str,
    max_tokens: int = 256,
) -> pd.DataFrame:
    """
    Run behavioral evaluation for all 4 API models with resume logic.
    Appends to existing behavioral_results.parquet.
    """
    ensure_dirs()

    if _BEHAVIORAL_PATH.exists():
        existing = pd.read_parquet(_BEHAVIORAL_PATH)
        logger.info("Loaded %d existing results for resume.", len(existing))
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple] = set()
    if len(existing) > 0 and "success_flag" in existing.columns:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))

    all_rows: list[pd.DataFrame] = [existing] if len(existing) > 0 else []

    for model_cfg in API_MODELS:
        logger.info("API evaluation: model=%s ...", model_cfg["name"])
        result_df = evaluate_api_model(model_cfg, pentad_df, run_id, completed_keys, max_tokens)
        if len(result_df) > 0:
            all_rows.append(result_df)
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_parquet(_BEHAVIORAL_PATH, index=False)
            logger.info("  Saved %d new rows for %s.", len(result_df), model_cfg["name"])

    final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if len(final) > 0:
        final = (
            final.sort_values("timestamp_utc")
            .drop_duplicates(subset=["prompt_id", "model_name", "sample_index"], keep="last")
            .reset_index(drop=True)
        )

        # Integrity check
        dupes = final[final.duplicated(subset=["prompt_id", "model_name", "sample_index"], keep=False)]
        if len(dupes) > 0:
            raise RuntimeError(
                f"Duplicate (prompt_id, model_name, sample_index) triples found: {len(dupes)} rows. "
                "Investigation required."
            )

        final.to_parquet(_BEHAVIORAL_PATH, index=False)

    logger.info("API behavioral evaluation complete. Total rows: %d", len(final))
    return final
