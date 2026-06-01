"""
Shared JSON response parsing for OSM and API behavioral evaluation.
"""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(
    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_RATIONALE_RE = re.compile(
    r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"',
    re.DOTALL,
)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def repair_json(raw: str) -> dict | None:
    """Attempt deterministic repair of a near-valid JSON string."""
    raw = raw.strip()
    if raw.startswith("{") and not raw.endswith("}"):
        raw = raw + "}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _strip_code_fence(raw: str) -> str:
    match = _FENCE_RE.search(raw)
    return match.group(1).strip() if match else raw.strip()


def _extract_fields_regex(raw: str) -> dict | None:
    """Last-resort field extraction when json.loads fails."""
    answer_m = _ANSWER_RE.search(raw)
    if not answer_m:
        return None
    conf_m = _CONFIDENCE_RE.search(raw)
    rat_m = _RATIONALE_RE.search(raw)
    return {
        "answer": answer_m.group(1).replace('\\"', '"'),
        "confidence": float(conf_m.group(1)) if conf_m else 0.0,
        "rationale": rat_m.group(1).replace('\\"', '"') if rat_m else "",
    }


def _parse_quoted_lines(raw: str) -> dict | None:
    """
    Parse Qwen-style multi-line outputs that omit JSON braces, e.g.:

        "C" Can't answer
        "0.7"
        "Insufficient information ..."
    """
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return None

    def _unquote(value: str) -> str:
        value = value.strip()
        if value.startswith('"') and value.endswith('"') and len(value) >= 2:
            return value[1:-1]
        return value

    if len(lines) == 1:
        only = _unquote(lines[0])
        if only:
            return {"answer": only, "confidence": 0.0, "rationale": ""}
        return None

    if len(lines) < 2:
        return None

    answer_parts: list[str] = []
    first = lines[0]
    match = re.match(r'^"([^"]*)"(.*)$', first)
    if match:
        if match.group(1):
            answer_parts.append(match.group(1))
        trailing = match.group(2).strip().strip('"')
        if trailing:
            answer_parts.append(trailing)
    else:
        answer_parts.append(_unquote(first))

    answer = " ".join(answer_parts).strip()
    if not answer:
        return None

    confidence = 0.0
    try:
        confidence = float(_unquote(lines[1]))
    except (TypeError, ValueError):
        pass

    rationale = _unquote(lines[2]) if len(lines) >= 3 else ""
    return {"answer": answer, "confidence": confidence, "rationale": rationale}


def parse_model_response(raw_response: str) -> tuple[bool, str, float, str, str, str]:
    """
    Parse a raw model response string into result fields.

    Returns
    -------
    (success_flag, parsed_answer, parsed_confidence, parsed_rationale,
     parse_method, failure_reason)
    """
    if not raw_response or not str(raw_response).strip():
        return False, "", 0.0, "", "failed", "empty_response"

    raw = _strip_code_fence(str(raw_response))

    candidate = raw if raw.startswith("{") else None
    if candidate is None:
        start = raw.find("{")
        if start != -1:
            candidate = raw[start:]

    parsed = None
    parse_method = "json"
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = repair_json(candidate)

    if parsed is None:
        parsed = _parse_quoted_lines(raw)
        if parsed is not None:
            parse_method = "quoted_lines"

    if parsed is None:
        parsed = _extract_fields_regex(raw)
        if parsed is not None:
            parse_method = "regex"

    if parsed is None:
        return False, "", 0.0, "", "failed", "parse_error"

    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        return False, "", 0.0, "", "failed", "parse_error"

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return (
        True,
        answer,
        confidence,
        str(parsed.get("rationale", "")),
        parse_method,
        "",
    )
