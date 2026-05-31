"""
File: CPU_Only/api_clients/bedrock_client.py
Purpose: AWS Bedrock client with OpenRouter fallback for API-1 (gpt-oss-20b)
         and API-2 (amazon.nova-2-lite-v1:0).

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - AWS Bedrock documentation: https://docs.aws.amazon.com/bedrock/

Retry/fallback policy (per spec Section 4.2):
  1. Bedrock primary, retry once.
  2. OpenRouter round-robin (2 keys, 2 attempts).
  3. Skip row and flag on all four failures.

Part of the MIRAGE codebase. See README.md for full project context.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, OPENROUTER_API_BASE_URL, OPENROUTER_KEYS, RESEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_BEDROCK_REGION = "us-east-1"

_OPENROUTER_MODEL_MAP = {
    "openai.gpt-oss-20b-1:0": "openai/gpt-4o-mini",
    "us.amazon.nova-2-lite-v1:0": "amazon/nova-lite-v1",
}


def _call_bedrock(model_id: str, messages: list[dict], max_tokens: int) -> str | None:
    """
    Call AWS Bedrock via the Converse API (model-agnostic).
    Guardrails are intentionally not attached — bias-audit benchmarks contain
    stereotyped language by design and must not be blocked.
    Returns text or None on failure.
    """
    try:
        import boto3  # type: ignore

        client = boto3.client(
            "bedrock-runtime",
            region_name=_BEDROCK_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )

        # Separate system message from conversation turns
        system_parts = [m for m in messages if m.get("role") == "system"]
        system_text = system_parts[0]["content"] if system_parts else RESEARCH_SYSTEM_PROMPT
        converse_msgs = [
            {"role": m["role"], "content": [{"text": m.get("content", "")}]}
            for m in messages if m.get("role") != "system"
        ]

        response = client.converse(
            modelId=model_id,
            system=[{"text": system_text}],
            messages=converse_msgs,
            inferenceConfig={"maxTokens": max_tokens},
            # No guardrailConfig — guardrails are opt-in; omitting means no filtering
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as exc:
        logger.warning("Bedrock call failed (model=%s): %s", model_id, exc)
        return None


def _call_openrouter(model_id: str, messages: list[dict], key: str, max_tokens: int) -> str | None:
    """Call OpenRouter as fallback. Returns text or None."""
    try:
        from openai import OpenAI

        or_model = _OPENROUTER_MODEL_MAP.get(model_id, model_id)
        client = OpenAI(api_key=key, base_url=OPENROUTER_API_BASE_URL, timeout=_TIMEOUT)
        response = client.chat.completions.create(
            model=or_model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            timeout=_TIMEOUT,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("OpenRouter fallback failed (model=%s): %s", model_id, exc)
        return None


def call_bedrock_with_fallback(
    model_id: str,
    messages: list[dict],
    max_tokens: int = 256,
) -> dict[str, Any]:
    """
    Call Bedrock with OpenRouter fallback per spec Section 4.2.

    Returns
    -------
    dict with keys:
        raw_response, route_used, key_index, attempt_count, success_flag, failure_reason
    """
    attempt_count = 0
    raw_response = ""
    route_used = "bedrock"

    # Attempt 1 + 2: Bedrock primary
    for _ in range(2):
        attempt_count += 1
        t0 = time.monotonic()
        raw = _call_bedrock(model_id, messages, max_tokens)
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "bedrock",
                "key_index": 0,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    # Attempts 3 + 4: OpenRouter round-robin
    _or_idx = 0
    for _ in range(2):
        attempt_count += 1
        key = OPENROUTER_KEYS[_or_idx % len(OPENROUTER_KEYS)]
        key_index = _or_idx % len(OPENROUTER_KEYS)
        _or_idx += 1
        t0 = time.monotonic()
        raw = _call_openrouter(model_id, messages, key, max_tokens)
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "openrouter",
                "key_index": key_index,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    # All four attempts failed
    return {
        "raw_response": "",
        "route_used": "failed",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": 0,
    }
