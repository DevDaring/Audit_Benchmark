"""
File: CPU_Only/judge_router.py
Purpose: Optional judge / answer extraction for malformed JSON responses.
         Cascades DeepSeek -> Mistral -> OpenRouter, two keys round-robined per
         cross-provider fallback -- if the chosen provider fails, returns None.

         DeepSeek is the default. The Gemini route was removed entirely: its keys
         were rate-limited through whole passes, so it added latency without ever
         returning a repair.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Wang et al. (2025). "Fairness through Difference Awareness." ACL 2025.

Part of the audit codebase (diagnosis half of CURE).
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_JUDGE_MODEL_NAME,
    DEEPSEEK_KEYS,
    MISTRAL_KEYS,
    MISTRAL_MODEL_NAME,
    OPENROUTER_API_BASE_URL,
    OPENROUTER_KEYS,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60
_JUDGE_SYSTEM = (
    "You are a JSON repair assistant. The following text is a malformed or incomplete "
    "response from a language model. Extract ONLY what is actually present: the answer, "
    "confidence, and rationale. If the text is truncated, empty, or contains no clear "
    'answer, set "answer" to an empty string "". Never invent, guess, or add an answer '
    "that is not present in the text. "
    'Return ONLY valid JSON: {"answer": "<answer or empty>", "confidence": <0.0-1.0>, "rationale": "<one sentence>"}'
)



def _call_deepseek_judge(raw_response: str) -> dict | None:
    """Use DeepSeek as judge. Round-robin keys."""
    from openai import OpenAI

    for i, key in enumerate(DEEPSEEK_KEYS):
        try:
            client = OpenAI(api_key=key, base_url=DEEPSEEK_API_BASE_URL, timeout=_TIMEOUT)
            response = client.chat.completions.create(
                model=DEEPSEEK_JUDGE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": raw_response},
                ],
                response_format={"type": "json_object"},
                max_tokens=256,
                timeout=_TIMEOUT,
            )
            raw = response.choices[0].message.content or ""
            return json.loads(raw)
        except Exception as exc:
            logger.warning("DeepSeek judge attempt %d failed: %s", i + 1, exc)
    return None


def _call_mistral_judge(raw_response: str) -> dict | None:
    """Use Mistral as judge. Round-robin keys."""
    from CPU_Only.api_clients.mistral_client import call_mistral_with_roundrobin

    result = call_mistral_with_roundrobin(
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": raw_response},
        ],
        max_tokens=256,
        model_name="mistral-small-latest",
    )
    if not result["success_flag"]:
        return None
    try:
        return json.loads(result["raw_response"])
    except json.JSONDecodeError:
        return None


def _call_openrouter_judge(raw_response: str) -> dict | None:
    """Use OpenRouter as the last-resort judge. Round-robin keys."""
    import os

    from openai import OpenAI

    model = os.getenv("OPENROUTER_PRIMARY_MODEL_NAME", "openai/gpt-4o-mini")
    for i, key in enumerate(OPENROUTER_KEYS):
        try:
            client = OpenAI(api_key=key, base_url=OPENROUTER_API_BASE_URL, timeout=_TIMEOUT)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": raw_response},
                ],
                max_tokens=256,
                timeout=_TIMEOUT,
            )
            return json.loads(response.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenRouter judge key %d failed: %s", i + 1, str(exc)[:160])
    return None


def judge(raw_response: str, provider: str = "deepseek") -> tuple[dict | None, str]:
    """
    Attempt to extract structured answer from a malformed raw response.

    Parameters
    ----------
    raw_response : str
        Raw model output that failed deterministic JSON parsing.
    provider : str
        'deepseek' (default) | 'mistral'. No automatic fallback between providers.

    Returns
    -------
    tuple[dict | None, str]
        (parsed_dict_or_None, parse_method_string)
    """
    # Cascade: DeepSeek -> Mistral -> OpenRouter, both keys round-robined inside each
    # tier before moving on, so the full path is six attempts. A single provider can be
    # forced by name; "auto" walks the cascade.
    tiers = [
        ("deepseek", _call_deepseek_judge, "judge_deepseek"),
        ("mistral", _call_mistral_judge, "judge_mistral"),
        ("openrouter", _call_openrouter_judge, "judge_openrouter"),
    ]
    known = {name for name, _, _ in tiers}
    if provider not in known and provider != "auto":
        raise ValueError(
            f"Unknown judge provider: '{provider}'. Use 'auto', or one of {sorted(known)}."
        )

    ordered = tiers if provider == "auto" else (
        [t for t in tiers if t[0] == provider] + [t for t in tiers if t[0] != provider]
    )

    for name, fn, method in ordered:
        result = fn(raw_response)
        if result is not None:
            return result, method
        logger.warning("Judge tier '%s' exhausted its keys; falling through.", name)

    logger.warning("All judge tiers exhausted; returning None.")
    return None, "judge_failed"
