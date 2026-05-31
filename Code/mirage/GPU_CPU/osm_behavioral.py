"""
File: GPU_CPU/osm_behavioral.py
Purpose: Behavioral evaluation of all 4 OSM models across the full pentad
         probe set. Produces behavioral_results.parquet.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - outlines: constrained JSON decoding. https://github.com/outlines-dev/outlines

Part of the MIRAGE codebase. See README.md for full project context.
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, RESEARCH_SYSTEM_PROMPT, ensure_dirs

logger = logging.getLogger(__name__)

_BEHAVIORAL_PATH = RESULTS_DIR / "behavioral_results.parquet"

_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["answer", "confidence", "rationale"],
}

_SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT


def _repair_json(raw: str) -> dict | None:
    """Attempt deterministic repair of a near-valid JSON string."""
    raw = raw.strip()
    if raw.startswith("{") and not raw.endswith("}"):
        raw = raw + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _generate_constrained(
    model: Any,
    tokenizer: Any,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """
    Generate a response using outlines constrained JSON decoding.
    Falls back to unconstrained generation if outlines fails.
    """
    try:
        import outlines  # type: ignore
        import outlines.models as om
        import outlines.generate as og

        gen = og.json(model, _JSON_SCHEMA)
        result = gen(prompt, max_tokens=max_tokens, temperature=temperature)
        return json.dumps(result)
    except Exception as exc:
        logger.debug("outlines constrained decode failed (%s), falling back.", exc)
        # Unconstrained fallback
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _build_prompt(system: str, user: str, tokenizer: Any) -> str:
    """Build a chat-formatted prompt."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"


def evaluate_osm_model(
    model_cfg: dict,
    model: Any,
    tokenizer: Any,
    pentad_df: pd.DataFrame,
    run_id: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    sample_index: int = 0,
) -> pd.DataFrame:
    """
    Evaluate a single OSM model on the full pentad dataset.

    Returns
    -------
    pd.DataFrame
        Result rows conforming to the MIRAGE result schema.
    """
    from GPU_CPU.utils_attention import _get_token_position  # noqa: F401

    model_name = model_cfg["name"]
    model_provider = "hf"
    try:
        model_version = model.config._name_or_path
    except Exception:
        model_version = model_cfg["hf_id"]

    rows: list[dict] = []
    total = len(pentad_df)

    for i, (_, prow) in enumerate(pentad_df.iterrows()):
        prompt_id = prow["prompt_id"]
        slot = prow.get("slot", "")

        # FM4 variance pass (sample_index > 0) must run on slot-a only.
        # Running it on all slots would multiply GPU work by ~12x and produce
        # variance data for slots that are not scored under FM4.
        if sample_index > 0 and slot != "a":
            continue

        prompt_text = str(prow.get("prompt_text", ""))
        if not prompt_text.strip():
            continue

        gold_answer = str(prow.get("gold_answer", ""))
        formatted = _build_prompt(_SYSTEM_PROMPT, prompt_text, tokenizer)
        t_start = time.monotonic()

        parse_method = "json"
        success_flag = True
        failure_reason = ""
        parsed_answer = ""
        parsed_confidence = 0.0
        parsed_rationale = ""
        raw_response = ""

        try:
            raw_response = _generate_constrained(model, tokenizer, formatted, temperature, max_tokens)
            parsed = json.loads(raw_response) if raw_response.startswith("{") else _repair_json(raw_response)
            if parsed is None:
                parse_method = "failed"
                success_flag = False
                failure_reason = "parse_error"
            else:
                parsed_answer = str(parsed.get("answer", ""))
                parsed_confidence = float(parsed.get("confidence", 0.0))
                parsed_rationale = str(parsed.get("rationale", ""))
        except Exception as exc:
            raw_response = str(exc)
            parse_method = "failed"
            success_flag = False
            failure_reason = "parse_error"

        latency_ms = int((time.monotonic() - t_start) * 1000)

        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed_id": prow.get("seed_id", ""),
                "seed_source": prow.get("seed_source", ""),
                "seed_category": prow.get("seed_category", ""),
                "seed_subcategory": prow.get("seed_subcategory", ""),
                "prompt_id": prompt_id,
                "slot": slot,
                "subvariant": prow.get("subvariant", ""),
                "gold_answer": gold_answer,
                "model_name": model_name,
                "model_provider": model_provider,
                "model_version": model_version,
                "route_used": "local",
                "key_index": -1,
                "attempt_count": 1,
                "prompt_text": prompt_text,
                "raw_response": raw_response,
                "parsed_answer": parsed_answer,
                "parsed_confidence": parsed_confidence,
                "parsed_rationale": parsed_rationale,
                "parse_method": parse_method,
                "success_flag": success_flag,
                "failure_reason": failure_reason,
                "latency_ms": latency_ms,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "sample_index": sample_index,
            }
        )

        if (i + 1) % 50 == 0:
            logger.info(
                "OSM %s: %d/%d prompts done (sample_index=%d).", model_name, i + 1, total, sample_index
            )

    return pd.DataFrame(rows)


def run_osm_behavioral(
    pentad_df: pd.DataFrame,
    models: dict[str, tuple[Any, Any]],
    run_id: str,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run behavioral evaluation for all OSM models.
    Uses resume logic: skips already-completed rows.
    """
    ensure_dirs()

    # Load existing results
    if _BEHAVIORAL_PATH.exists():
        existing = pd.read_parquet(_BEHAVIORAL_PATH)
        logger.info("Loaded %d existing behavioral results.", len(existing))
    else:
        existing = pd.DataFrame()

    completed_keys: set[tuple] = set()
    if len(existing) > 0:
        for _, row in existing[existing["success_flag"] == True].iterrows():  # noqa: E712
            completed_keys.add((row["prompt_id"], row["model_name"], int(row["sample_index"])))

    all_rows: list[pd.DataFrame] = [existing] if len(existing) > 0 else []

    for model_cfg in OSM_MODELS:
        model_name = model_cfg["name"]
        if model_name not in models:
            logger.warning("Model '%s' not loaded, skipping.", model_name)
            continue

        model, tokenizer = models[model_name]

        # Deterministic pass (temperature=0, sample_index=0)
        missing_det = pentad_df[
            ~pentad_df["prompt_id"].apply(
                lambda pid: (pid, model_name, 0) in completed_keys
            )
        ]
        if len(missing_det) > 0:
            logger.info(
                "OSM %s: deterministic pass on %d prompts ...", model_name, len(missing_det)
            )
            det_results = evaluate_osm_model(
                model_cfg, model, tokenizer, missing_det, run_id, temperature=0.0, sample_index=0
            )
            all_rows.append(det_results)
            # Incremental write
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_parquet(_BEHAVIORAL_PATH, index=False)
            logger.info("  Saved deterministic results incrementally.")

        # Variance pass (temperature=0.7, sample_index=1-5)
        for si in range(1, 6):
            missing_var = pentad_df[
                ~pentad_df["prompt_id"].apply(
                    lambda pid: (pid, model_name, si) in completed_keys
                )
            ]
            if len(missing_var) > 0:
                logger.info(
                    "OSM %s: variance pass sample_index=%d on %d prompts ...",
                    model_name, si, len(missing_var),
                )
                var_results = evaluate_osm_model(
                    model_cfg, model, tokenizer, missing_var, run_id,
                    temperature=0.7, sample_index=si
                )
                all_rows.append(var_results)
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_parquet(_BEHAVIORAL_PATH, index=False)

    # Final deduplication
    final = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if len(final) > 0:
        final = (
            final.sort_values("timestamp_utc")
            .drop_duplicates(subset=["prompt_id", "model_name", "sample_index"], keep="last")
            .reset_index(drop=True)
        )
        final.to_parquet(_BEHAVIORAL_PATH, index=False)

    logger.info("OSM behavioral evaluation complete. Total rows: %d", len(final))
    return final
