"""
File: CPU_Only/api_clients/mistral_client.py
Purpose: Mistral platform client with 2-key round-robin for API-4.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Mistral AI Python SDK: https://github.com/mistralai/client-python

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
import time
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import MISTRAL_KEYS, MISTRAL_MODEL_NAME

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_MAX_ATTEMPTS_PER_KEY = 2


class _MistralRoundRobin:
    def __init__(self) -> None:
        self._idx = 0

    def next(self) -> tuple[str, int]:
        key = MISTRAL_KEYS[self._idx % len(MISTRAL_KEYS)]
        idx = self._idx % len(MISTRAL_KEYS)
        self._idx += 1
        return key, idx


_rr = _MistralRoundRobin()


def _call_mistral(key: str, model_name: str, messages: list[dict], max_tokens: int) -> str | None:
    """Single Mistral API call. Returns text or None."""
    try:
        from mistralai.client import Mistral  # type: ignore

        client = Mistral(api_key=key)
        response = client.chat.complete(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            safe_prompt=False,  # do not inject Mistral's safety preamble
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning("Mistral call failed: %s", exc)
        return None


def call_mistral_with_roundrobin(
    messages: list[dict],
    max_tokens: int = 256,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Call Mistral with round-robin key rotation.
    Skips and flags if all keys exhausted.
    """
    if model_name is None:
        model_name = MISTRAL_MODEL_NAME

    attempt_count = 0
    t0 = time.monotonic()

    for _ in range(len(MISTRAL_KEYS) * _MAX_ATTEMPTS_PER_KEY):
        key, key_index = _rr.next()
        attempt_count += 1
        raw = _call_mistral(key, model_name, messages, max_tokens)
        if raw is not None:
            return {
                "raw_response": raw,
                "route_used": "mistral",
                "key_index": key_index,
                "attempt_count": attempt_count,
                "success_flag": True,
                "failure_reason": "",
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }

    return {
        "raw_response": "",
        "route_used": "mistral",
        "key_index": -1,
        "attempt_count": attempt_count,
        "success_flag": False,
        "failure_reason": "api_error",
        "latency_ms": int((time.monotonic() - t0) * 1000),
    }
