"""
File: GPU_CPU/osm_behavioral.py
Purpose: Behavioral evaluation of all 4 OSM models across the full pentad
         probe set. Produces behavioral_results.parquet.

Parallelism strategy (A100 80 GB SXM4):
  - All 4 OSM models are kept in VRAM simultaneously (~56 GB total), so there
    is no model-reload overhead between evaluation phases.
  - Batch inference: EVAL_BATCH_SIZE prompts are tokenised and forwarded in a
    single GPU call instead of one at a time.  On 80 GB with 24 GB headroom
    a batch of 8 fits comfortably even for the largest model (Phi-4-mini, ~8 GB).
  - Outlines constrained decoding is single-prompt only; the batch path uses
    raw model.generate() with left-padding, which is equally deterministic at
    temperature=0.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, RESEARCH_SYSTEM_PROMPT, ensure_dirs

logger = logging.getLogger(__name__)

# Number of prompts to forward in a single batched GPU call.
# 8 fits in 80 GB for all models; lower to 4 if OOM errors appear on smaller GPUs.
EVAL_BATCH_SIZE: int = 8

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
    Used for single-prompt paths (e.g. when batch_size=1).
    """
    try:
        import outlines  # type: ignore
        import outlines.generate as og

        gen = og.json(model, _JSON_SCHEMA)
        result = gen(prompt, max_tokens=max_tokens, temperature=temperature)
        return json.dumps(result)
    except Exception as exc:
        logger.debug("outlines constrained decode failed (%s), falling back.", exc)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _generate_constrained_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Batch generation path.  Tokenises all prompts together with left-padding
    and runs a single model.generate() call, returning one decoded string per
    prompt.

    Outlines does not support batching, so this path always uses the raw
    model.generate() fallback.  At temperature=0 the output is fully
    deterministic, identical to the single-prompt unconstrained fallback.

    Parameters
    ----------
    prompts : list[str]
        Up to EVAL_BATCH_SIZE formatted prompt strings.

    Returns
    -------
    list[str]
        Decoded output strings, one per input prompt (same order).
    """
    import torch

    if not prompts:
        return []

    # Left-padding so all sequences in the batch end at the same position —
    # this is required for decoder-only causal models.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    try:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        tokenizer.padding_side = orig_padding_side

    input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(out[i][input_len:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]


def _build_prompt(system: str, user: str, tokenizer: Any) -> str:
    """Build a chat-formatted prompt."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"


def _parse_raw_response(raw_response: str) -> tuple[bool, str, float, str, str, str]:
    """
    Parse a raw model response string into result fields.

    Returns
    -------
    (success_flag, parsed_answer, parsed_confidence, parsed_rationale,
     parse_method, failure_reason)
    """
    if not raw_response.strip():
        return False, "", 0.0, "", "failed", "empty_response"

    candidate = raw_response if raw_response.startswith("{") else None
    if candidate is None:
        # Try to find the first JSON object in the string
        start = raw_response.find("{")
        if start != -1:
            candidate = raw_response[start:]

    parsed = None
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = _repair_json(candidate)

    if parsed is None:
        return False, "", 0.0, "", "failed", "parse_error"

    return (
        True,
        str(parsed.get("answer", "")),
        float(parsed.get("confidence", 0.0)),
        str(parsed.get("rationale", "")),
        "json",
        "",
    )


def evaluate_osm_model(
    model_cfg: dict,
    model: Any,
    tokenizer: Any,
    pentad_df: pd.DataFrame,
    run_id: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    sample_index: int = 0,
    batch_size: int = EVAL_BATCH_SIZE,
) -> pd.DataFrame:
    """
    Evaluate a single OSM model on the full pentad dataset.

    Parallelism: prompts are processed in batches of `batch_size` using a
    single GPU forward pass per batch (via _generate_constrained_batch).
    At temperature=0 this is deterministically equivalent to single-prompt
    evaluation.

    Parameters
    ----------
    batch_size : int
        Number of prompts per GPU call.  Default is EVAL_BATCH_SIZE (8).
        Lower to 1 or 2 when debugging or on GPUs with <24 GB free VRAM.

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

    # Pre-filter: FM4 variance pass only needs slot-a.
    if sample_index > 0:
        pentad_df = pentad_df[pentad_df.get("slot", pd.Series(dtype=str)) == "a"].copy()

    # Drop empty prompts.
    pentad_df = pentad_df[pentad_df["prompt_text"].astype(str).str.strip() != ""].reset_index(drop=True)

    rows: list[dict] = []
    total = len(pentad_df)
    now_utc = datetime.now(timezone.utc).isoformat()

    for batch_start in range(0, total, batch_size):
        batch = pentad_df.iloc[batch_start : batch_start + batch_size]

        # Build formatted prompts for the whole batch.
        formatted_prompts = [
            _build_prompt(_SYSTEM_PROMPT, str(r["prompt_text"]), tokenizer)
            for _, r in batch.iterrows()
        ]

        t_start = time.monotonic()
        try:
            raw_responses = _generate_constrained_batch(
                model, tokenizer, formatted_prompts, temperature, max_tokens
            )
        except Exception as exc:
            # If batch generation fails entirely, produce error rows for all.
            logger.warning(
                "OSM %s: batch generation failed (%s). Falling back to single-prompt.", model_name, exc
            )
            raw_responses = []
            for fp in formatted_prompts:
                try:
                    raw_responses.append(
                        _generate_constrained(model, tokenizer, fp, temperature, max_tokens)
                    )
                except Exception as e2:
                    raw_responses.append(str(e2))

        latency_ms_total = int((time.monotonic() - t_start) * 1000)
        per_prompt_ms = latency_ms_total // max(len(batch), 1)

        for j, (_, prow) in enumerate(batch.iterrows()):
            raw_response = raw_responses[j] if j < len(raw_responses) else ""
            success_flag, parsed_answer, parsed_confidence, parsed_rationale, parse_method, failure_reason = (
                _parse_raw_response(raw_response)
            )

            rows.append(
                {
                    "run_id": run_id,
                    "timestamp_utc": now_utc,
                    "seed_id": prow.get("seed_id", ""),
                    "seed_source": prow.get("seed_source", ""),
                    "seed_category": prow.get("seed_category", ""),
                    "seed_subcategory": prow.get("seed_subcategory", ""),
                    "prompt_id": prow["prompt_id"],
                    "slot": prow.get("slot", ""),
                    "subvariant": prow.get("subvariant", ""),
                    "gold_answer": str(prow.get("gold_answer", "")),
                    "model_name": model_name,
                    "model_provider": model_provider,
                    "model_version": model_version,
                    "route_used": "local",
                    "key_index": -1,
                    "attempt_count": 1,
                    "prompt_text": str(prow.get("prompt_text", "")),
                    "raw_response": raw_response,
                    "parsed_answer": parsed_answer,
                    "parsed_confidence": parsed_confidence,
                    "parsed_rationale": parsed_rationale,
                    "parse_method": parse_method,
                    "success_flag": success_flag,
                    "failure_reason": failure_reason,
                    "latency_ms": per_prompt_ms,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "sample_index": sample_index,
                }
            )

        if (batch_start + len(batch)) % 50 < batch_size:
            logger.info(
                "OSM %s: %d/%d prompts done (sample_index=%d, batch_size=%d).",
                model_name, batch_start + len(batch), total, sample_index, batch_size,
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
