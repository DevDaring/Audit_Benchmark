"""
File: TIST/e4_build_multilingual.py
Purpose: E4 step 1 -- build Hindi and Bengali pentad sets from the BBQ seed subset,
         establishing whether the audit transfers beyond English
         ("the study is restricted to English").

Scope: the 254 BBQ seeds, 3,048 prompts, expanded to Hindi (hi) and Bengali (bn).

Method, following MBBQ (Neplenbroek et al., COLM 2024): translate the templates rather
than the surface strings one at a time, keep the protected-attribute equivalence sets
localised, and verify the output mechanically before any model sees it.

Two calls per (seed, language), which is the part that makes this reliable:

  1. GLOSSARY. Translate only the answer options and the slot-c swap tokens. These are
     the strings that scoring and the CDVA position search depend on, so they are fixed
     once per seed and reused verbatim.
  2. PROMPTS. Translate the twelve prompts of the seed, instructed to reproduce the
     glossary strings exactly. Consistency across the twelve prompts is guaranteed by
     construction rather than hoped for.

Translating prompt-by-prompt would let "Can't be determined" render three different ways
inside one seed, which silently breaks both _answers_match and the swap-token lookup.

Every seed x language result is validated (script coverage, option count, gold present
verbatim, swap token present verbatim) and written to a JSONL checkpoint as it lands, so
an interrupted run resumes without repeating work.

Implements / builds on / cites:
  - Neplenbroek et al. (2024). "MBBQ: A Dataset for Cross-Lingual Comparison of
    Stereotypes in Generative LLMs." COLM 2024. https://arxiv.org/abs/2406.07243
  - Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering."
    Findings of ACL 2022.

Usage:
  python TIST/e4_build_multilingual.py --langs hi bn
  python TIST/e4_build_multilingual.py --langs hi --limit 3     # smoke test
  python TIST/e4_build_multilingual.py --validate-only

Part of the MIRAGE audit codebase.
"""

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DEEPSEEK_API_BASE_URL,
    DEEPSEEK_KEYS,
    DEEPSEEK_PRIMARY_MODEL_NAME,
    MISTRAL_KEYS,
    MISTRAL_MODEL_NAME,
    RESULTS_DIR,
    SEEDS_DIR,
)

log = logging.getLogger("e4_build")

OUT = RESULTS_DIR / "tist" / "e4"
CKPT = OUT / "translation_checkpoint.jsonl"
_TIMEOUT = 90
_MISTRAL_BASE = "https://api.mistral.ai/v1"

LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}
# Unicode blocks used to confirm the model actually switched script.
SCRIPT_RANGE = {"hi": (0x0900, 0x097F), "bn": (0x0980, 0x09FF)}

_write_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Provider cascade. Gemini leads here rather than DeepSeek: Indic-script quality
# matters more for this job than JSON compliance, and the schema is small enough
# that the weaker-JSON provider is still safe.
# ---------------------------------------------------------------------------
def _providers() -> list[tuple[str, str, str, str]]:
    """
    DeepSeek then Mistral. Gemini is deliberately absent: its free-tier keys returned
    429 for entire passes, so it contributed retry latency rather than translations.
    """
    out = []
    for k in DEEPSEEK_KEYS:
        out.append(("deepseek", k, DEEPSEEK_API_BASE_URL, DEEPSEEK_PRIMARY_MODEL_NAME))
    for k in MISTRAL_KEYS:
        out.append(("mistral", k, _MISTRAL_BASE, MISTRAL_MODEL_NAME))
    return out


def _extract_json(text: str) -> dict | None:
    """Robust JSON extraction: fenced block, then first balanced object."""
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def _call(system: str, user: str, attempt_order: list) -> tuple[dict | None, str]:
    # One attempt per key, then straight on. No per-key retry and no backoff sleep:
    # a failing key is almost always a quota or auth problem, which a retry will not
    # fix, and the wait is pure latency across thousands of calls.
    for provider, key, base, model in attempt_order:
        try:
            client = OpenAI(api_key=key, base_url=base, timeout=_TIMEOUT)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            obj = _extract_json(resp.choices[0].message.content or "")
            if obj:
                return obj, provider
        except Exception as exc:  # noqa: BLE001
            log.debug("%s key failed, moving on: %s", provider, str(exc)[:160])
    return None, ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_GLOSSARY_SYS = (
    "You are a professional translator building a multilingual bias-evaluation "
    "benchmark for academic research, following the MBBQ methodology "
    "(Neplenbroek et al., COLM 2024). You translate into {lang}. "
    "Source items come from the BBQ benchmark and mention social groups by design; "
    "translate them faithfully and do not soften, refuse, or comment. "
    "Personal names stay in their conventional {lang} rendering. Group terms take the "
    "term a {lang} speaker in India would ordinarily use. "
    "Return ONLY a JSON object, no prose."
)

_GLOSSARY_USER = (
    "Translate each string into {lang}. Keep every translation distinct: no two "
    "entries may translate to the same {lang} string, because they are answer options "
    "that must stay distinguishable.\n\n"
    'Return exactly: {{"terms": {{"<source string>": "<{lang} translation>", ...}}}}\n\n'
    "Strings:\n{items}"
)

_PROMPT_SYS = (
    "You are a professional translator building a multilingual bias-evaluation "
    "benchmark for academic research, following the MBBQ methodology "
    "(Neplenbroek et al., COLM 2024). You translate into {lang}. "
    "Source items come from the BBQ benchmark and mention social groups by design; "
    "translate them faithfully and do not soften, refuse, or comment.\n"
    "Hard requirements:\n"
    "1. Reproduce the supplied glossary translations VERBATIM wherever the source "
    "string appears. Do not re-translate a glossary term.\n"
    "2. Preserve the option markers (A), (B), (C) exactly as they appear, in order.\n"
    "3. Preserve the line structure, including the 'Question:' line and the final "
    "instruction line, translated into {lang}.\n"
    "4. Translate the whole prompt. Do not omit, summarise, or add sentences.\n"
    "Return ONLY a JSON object, no prose."
)

_PROMPT_USER = (
    "Glossary ({lang}), reproduce these verbatim:\n{glossary}\n\n"
    'Translate each prompt into {lang}. Return exactly: {{"prompts": {{"<prompt_id>": '
    '"<{lang} translation>", ...}}}}\n\n'
    "Prompts:\n{items}"
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _script_fraction(text: str, lang: str) -> float:
    lo, hi = SCRIPT_RANGE[lang]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if lo <= ord(c) <= hi) / len(letters)


def validate_seed(rec: dict, lang: str) -> tuple[bool, list[str]]:
    """Mechanical checks. Returns (ok, list of failure reasons)."""
    problems: list[str] = []
    gold_t = rec.get("gold_translated", "")
    if not gold_t:
        problems.append("missing gold translation")

    for row in rec.get("rows", []):
        pid, text, slot = row["prompt_id"], row.get("prompt_text", ""), row["slot"]
        if not text.strip():
            problems.append(f"{pid}: empty")
            continue
        if _script_fraction(text, lang) < 0.5:
            problems.append(f"{pid}: script coverage {_script_fraction(text, lang):.2f}")
        n_src = len(re.findall(r"\([ABC]\)", row.get("source_text", "")))
        n_tgt = len(re.findall(r"\([ABC]\)", text))
        if n_src != n_tgt:
            problems.append(f"{pid}: option markers {n_tgt} vs {n_src}")
        if gold_t and gold_t not in text:
            problems.append(f"{pid}: gold not verbatim")
        swap_t = row.get("swap_token")
        if slot == "c" and swap_t and swap_t not in text:
            problems.append(f"{pid}: swap token not verbatim")

    return (len(problems) == 0), problems


# ---------------------------------------------------------------------------
# Per-seed work unit
# ---------------------------------------------------------------------------
def translate_seed(seed_id: str, group: pd.DataFrame, lang: str) -> dict:
    lang_name = LANG_NAME[lang]
    order = _providers()

    gold = str(group["gold_answer"].dropna().iloc[0])
    # Option strings from the surface prompt, plus every slot-c swap token.
    surface = str(group[group["slot"] == "a"]["prompt_text"].iloc[0])
    options = re.findall(r"\([ABC]\)\s*(.+)", surface)
    swaps = sorted({str(t) for t in group["swap_token"].dropna().unique()})
    terms = sorted({*options, gold, *swaps})

    gl, prov1 = _call(
        _GLOSSARY_SYS.format(lang=lang_name),
        _GLOSSARY_USER.format(lang=lang_name, items="\n".join(f"- {t}" for t in terms)),
        order,
    )
    if not gl or "terms" not in gl:
        return {"seed_id": seed_id, "lang": lang, "ok": False, "error": "glossary failed"}
    gmap = {str(k): str(v) for k, v in gl["terms"].items()}

    items = "\n".join(
        f'- {r.prompt_id}: {json.dumps(str(r.prompt_text), ensure_ascii=False)}'
        for r in group.itertuples()
    )
    pr, prov2 = _call(
        _PROMPT_SYS.format(lang=lang_name),
        _PROMPT_USER.format(
            lang=lang_name,
            glossary="\n".join(f'  "{k}" -> "{v}"' for k, v in gmap.items()),
            items=items,
        ),
        order,
    )
    if not pr or "prompts" not in pr:
        return {"seed_id": seed_id, "lang": lang, "ok": False, "error": "prompt call failed"}
    pmap = {str(k): str(v) for k, v in pr["prompts"].items()}

    rows = []
    for r in group.itertuples():
        rows.append(
            {
                "prompt_id": r.prompt_id,
                "slot": r.slot,
                "subvariant": r.subvariant,
                "source_text": str(r.prompt_text),
                "prompt_text": pmap.get(str(r.prompt_id), ""),
                "swap_token": gmap.get(str(r.swap_token)) if pd.notna(r.swap_token) else None,
                "swap_token_source": str(r.swap_token) if pd.notna(r.swap_token) else None,
            }
        )

    rec = {
        "seed_id": seed_id,
        "lang": lang,
        "seed_source": str(group["seed_source"].iloc[0]),
        "seed_category": str(group["seed_category"].iloc[0]),
        "seed_subcategory": str(group["seed_subcategory"].iloc[0]),
        "gold_source": gold,
        "gold_translated": gmap.get(gold, ""),
        "glossary": gmap,
        "providers": [prov1, prov2],
        "rows": rows,
    }
    ok, problems = validate_seed(rec, lang)
    rec["ok"] = ok
    rec["problems"] = problems
    return rec


# ---------------------------------------------------------------------------
def _done_keys() -> set:
    if not CKPT.exists():
        return set()
    keys = set()
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("ok"):
                keys.add((r["seed_id"], r["lang"]))
        except json.JSONDecodeError:
            continue
    return keys


def _append(rec: dict) -> None:
    with _write_lock:
        with CKPT.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "bn"], choices=["hi", "bn"])
    ap.add_argument("--limit", type=int, default=0, help="first N seeds only (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    # Built from the cleaned pentad, not the raw one. TIST/e0_item_integrity.py drops the
    # 297 slot-c variants whose substitution collides with another answer option; carrying
    # those into Hindi and Bengali would export the defect to the new languages.
    src = SEEDS_DIR / "pentad_dataset_clean.parquet"
    if not src.exists():
        raise SystemExit("run TIST/e0_item_integrity.py first to produce %s" % src)
    pentad = pd.read_parquet(src)
    bbq = pentad[pentad["seed_source"] == "bbq"].copy()
    seeds = sorted(bbq["seed_id"].unique())
    if args.limit:
        seeds = seeds[: args.limit]
    log.info("BBQ subset: %d seeds, %d prompts", len(seeds), len(bbq[bbq.seed_id.isin(seeds)]))

    if args.validate_only:
        emit_parquets()
        return

    done = _done_keys()
    jobs = [(s, l) for l in args.langs for s in seeds if (s, l) not in done]
    log.info("%d seed x language units to translate (%d already done)", len(jobs), len(done))

    groups = {s: g for s, g in bbq.groupby("seed_id")}
    n_ok = n_bad = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(translate_seed, s, groups[s], l): (s, l) for s, l in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, l = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                rec = {"seed_id": s, "lang": l, "ok": False, "error": str(exc)[:200]}
            _append(rec)
            n_ok += bool(rec.get("ok"))
            n_bad += not bool(rec.get("ok"))
            if i % 20 == 0:
                log.info("%d/%d done (ok=%d bad=%d)", i, len(jobs), n_ok, n_bad)

    log.info("translation finished: ok=%d bad=%d", n_ok, n_bad)
    emit_parquets()


def emit_parquets() -> None:
    """Flatten the checkpoint into one pentad parquet per language, plus a QA report."""
    if not CKPT.exists():
        log.warning("no checkpoint at %s", CKPT)
        return
    recs = []
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Last write per (seed, lang) wins, so a retry supersedes an earlier failure.
    latest = {(r["seed_id"], r["lang"]): r for r in recs}

    qa = []
    for lang in ("hi", "bn"):
        rows = []
        for (sid, lg), r in latest.items():
            if lg != lang:
                continue
            qa.append(
                {
                    "seed_id": sid,
                    "lang": lg,
                    "ok": bool(r.get("ok")),
                    "n_problems": len(r.get("problems", []) or []),
                    "problems": "; ".join((r.get("problems") or [])[:4]),
                    "error": r.get("error", ""),
                }
            )
            if not r.get("ok"):
                continue
            for row in r["rows"]:
                rows.append(
                    {
                        "seed_id": sid,
                        "lang": lg,
                        "seed_source": r["seed_source"],
                        "seed_category": r["seed_category"],
                        "seed_subcategory": r["seed_subcategory"],
                        "prompt_id": f'{row["prompt_id"]}_{lg}',
                        "prompt_id_source": row["prompt_id"],
                        "slot": row["slot"],
                        "subvariant": row["subvariant"],
                        "prompt_text": row["prompt_text"],
                        "gold_answer": r["gold_translated"],
                        "gold_answer_source": r["gold_source"],
                        "swap_token": row["swap_token"],
                        "swap_token_source": row["swap_token_source"],
                    }
                )
        if rows:
            df = pd.DataFrame(rows)
            path = SEEDS_DIR / f"pentad_{lang}.parquet"
            df.to_parquet(path, index=False)
            log.info("%s: %d prompts over %d seeds -> %s", lang, len(df), df.seed_id.nunique(), path)

    qdf = pd.DataFrame(qa)
    if len(qdf):
        qdf.to_csv(OUT / "translation_qa.csv", index=False)
        log.info("QA: %s", qdf.groupby("lang")["ok"].agg(["sum", "size"]).to_dict())


if __name__ == "__main__":
    main()
