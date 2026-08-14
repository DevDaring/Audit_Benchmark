"""
File: TIST/e4_api_multilingual.py
Purpose: Run the four closed-API models over the Hindi and Bengali pentads, gated by the
         same language-competence probe the open models pass through.

Runs on CPU only; every call goes to a vendor endpoint, so this belongs on the rented
server rather than on a GPU lease.

Order of work per (model, language):

  1. PROBE. Score the model on slot-a items in that language whose gold answer is a
     specific option rather than the "cannot be determined" escape. A model that answers
     by constant policy cannot pass. The verdict lands in results/tist/e4/competence.json,
     the same file the open models write, so the gate is one artefact for all eight.
  2. GATE. Below the competence floor, the pair is recorded as not-applicable with its
     measured accuracy and skipped. It is never reported as a bias failure or as a zero.
  3. RUN. Above it, evaluate the full pentad for that language.

Why the probe matters here as much as on the GPU side. Auditing bias in a language a model
cannot read measures comprehension while labelling the result as bias, which is the exact
construct-validity failure this paper documents in behavioural benchmarks. The open-model
run found five of eight pairs competent and three not, including two that vendor
documentation would have excluded and two it would have wrongly admitted.

Everything is resumable. Each (model, language) writes its own parquet, and a partially
finished language resumes from the rows already stored.

Usage, on the server:
  cd /home/Debz/Research/Audit_Benchmark/Code/audit
  /home/Debz/Research/tmp/venv/bin/python TIST/e4_api_multilingual.py --langs hi bn
  ... --models gemini-2.5-flash          # one model
  ... --probe-only                       # decide the gate without the full run
  ... --limit 20                         # smoke test

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_MODELS, RESULTS_DIR, SEEDS_DIR  # noqa: E402
from CPU_Only.api_behavioral import evaluate_api_model  # noqa: E402
from CPU_Only.scoring import _answers_match  # noqa: E402
from TIST.competence_probe import (  # noqa: E402
    MIN_ITEMS,
    evaluate,
    load,
    probe_items,
    record,
)
from TIST.language_support import skip_reason  # noqa: E402

log = logging.getLogger("e4_api")
OUT = RESULTS_DIR / "tist" / "e4"


def _out_path(lang: str, model_name: str) -> Path:
    return OUT / f"behav_{lang}_{model_name}.parquet"


def _score(df: pd.DataFrame) -> tuple[int, int]:
    """Correct and scorable counts, using the production answer comparison."""
    n_ok = scorable = 0
    for _, r in df.iterrows():
        gold = str(r.get("gold_answer", ""))
        if not gold:
            continue
        scorable += 1
        if _answers_match(str(r.get("parsed_answer", "")), gold, str(r.get("seed_source", ""))):
            n_ok += 1
    return n_ok, scorable


def _evaluate(model_cfg: dict, items: pd.DataFrame, existing: pd.DataFrame | None) -> pd.DataFrame:
    """Thin wrapper so probe and full run share one calling path."""
    completed: set = set()
    working = existing if existing is not None else pd.DataFrame()
    if existing is not None and len(existing):
        completed = set(zip(existing.get("prompt_id", []), existing.get("model_name", [])))
    return evaluate_api_model(
        model_cfg, items, run_id=str(uuid.uuid4()),
        completed_keys=completed, working=working,
    )


def run(models: list[dict], langs: list[str], probe_only: bool, limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    for cfg in models:
        name = cfg["name"]
        for lang in langs:
            src = SEEDS_DIR / f"pentad_{lang}.parquet"
            if not src.exists():
                log.warning("%s absent; skipping %s", src.name, lang)
                continue
            pen = pd.read_parquet(src)

            # ---- 1. probe, unless a verdict already exists -------------------
            # Re-probe when the stored verdict was itself untestable. A smoke test with
            # --limit writes an "only N scorable items" verdict, and trusting that would
            # permanently gate the pair out on evidence the probe itself called unusable.
            verdict = load(RESULTS_DIR).get(f"{name}|{lang}")
            if verdict is not None and (verdict.get("n_items") or 0) < MIN_ITEMS:
                log.info("discarding untestable verdict for %s / %s (n=%s); re-probing",
                         name, lang, verdict.get("n_items"))
                verdict = None
            if verdict is None:
                items = probe_items(pen)
                if limit:
                    items = items.head(limit)
                log.info("probe %s / %s on %d determinate items", name, lang, len(items))
                try:
                    df = _evaluate(cfg, items, None)
                except Exception as exc:  # noqa: BLE001
                    log.error("probe failed for %s / %s: %s", name, lang, str(exc)[:200])
                    continue
                n_ok, scorable = _score(df)
                verdict = evaluate(n_ok, scorable)
                verdict.update({"lang": lang, "model_name": name, "route": "api"})
                record(RESULTS_DIR, name, lang, verdict)

            if not verdict.get("competent"):
                log.info("SKIP %s / %s: %s", name, lang,
                         skip_reason(name, lang, RESULTS_DIR))
                continue
            log.info("GATE PASSED %s / %s: %s", name, lang, verdict.get("reason"))
            if probe_only:
                continue

            # ---- 2. full run, resumable -------------------------------------
            out = _out_path(lang, name)
            existing = None
            if out.exists():
                existing = pd.read_parquet(out)
                if len(existing) >= len(pen):
                    log.info("%s already complete (%d rows), skipping", out.name, len(existing))
                    continue
                log.info("%s resuming from %d of %d rows", out.name, len(existing), len(pen))

            items = pen.head(limit) if limit else pen
            log.info("running %s / %s on %d prompts", name, lang, len(items))
            try:
                df = _evaluate(cfg, items, existing)
            except Exception as exc:  # noqa: BLE001
                log.error("run failed for %s / %s: %s", name, lang, str(exc)[:250])
                continue
            df["lang"] = lang
            df.to_parquet(out, index=False)
            n_ok, scorable = _score(df)
            log.info("%s / %s done: %d rows, %d/%d correct on scorable items",
                     name, lang, len(df), n_ok, scorable)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["hi", "bn"], choices=["hi", "bn"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    models = [m for m in API_MODELS if not args.models or m["name"] in args.models]
    log.info("models: %s | langs: %s", [m["name"] for m in models], args.langs)
    run(models, args.langs, args.probe_only, args.limit)
    log.info("finished")


if __name__ == "__main__":
    main()
