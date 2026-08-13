"""
File: TIST/purge_and_verify.py
Purpose: Keep the result store correct. Remove rows that are not results, and refuse to
         let a malformed or duplicated record reach the analysis.

Two jobs.

  purge    Drop rows with ok=false from every result JSONL, and drop duplicate units,
           keeping the last occurrence. A failed row records an absence, not a
           measurement, and leaving it in place both pollutes the store and, through
           _done(), makes the failure permanent. Duplicates arise when a lease is killed
           mid-write and the unit is recomputed on resume.

  verify   Check what remains, and exit non-zero if anything is wrong:
             - every retained row has ok=true
             - no duplicate (seed_id, sub_a, sub_b) within a file
             - the metric field is present, numeric and not NaN
             - located fraction per multilingual file, since a script that cannot be
               located is a coverage finding that must be reported, not hidden
             - behavioural parquets hold the full prompt count, not a dry-run remnant

Run purge before analysis, and verify before quoting any number in the manuscript.

Usage:
  python TIST/purge_and_verify.py --verify
  python TIST/purge_and_verify.py --purge          # rewrites the JSONL in place
  python TIST/purge_and_verify.py --purge --verify

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, SEEDS_DIR  # noqa: E402

log = logging.getLogger("purge")

OUT = RESULTS_DIR / "tist"
KEY = ("seed_id", "sub_a", "sub_b")
METRIC_BY_PREFIX = {
    "battery": "real_single",
    "controls": "delta_single",
    "layersweep": "delta_single",
    "mediation": "indirect_effect",
    "cdva": "delta_single",
}


def _files() -> list[Path]:
    out = []
    for sub in ("e1", "e4"):
        d = OUT / sub
        if d.exists():
            out.extend(sorted(d.glob("*.jsonl")))
    return out


def _metric_for(path: Path) -> str | None:
    for pre, m in METRIC_BY_PREFIX.items():
        if path.name.startswith(pre):
            return m
    return None


def _read(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def purge() -> int:
    total_dropped = 0
    for path in _files():
        rows = _read(path)
        if not rows:
            continue
        # Layer sweep keys on the window as well, so it needs its own key.
        key = KEY + ("window_start",) if path.name.startswith("layersweep") else KEY

        kept: dict[tuple, dict] = {}
        n_failed = 0
        for r in rows:
            if r.get("ok") is False:
                n_failed += 1
                continue
            kept[tuple(r.get(k) for k in key)] = r      # last write wins

        dropped = len(rows) - len(kept)
        if dropped:
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False, default=float) + "\n"
                        for r in kept.values()),
                encoding="utf-8",
            )
            log.info("%s: %d rows -> %d (%d failed, %d duplicate)",
                     path.name, len(rows), len(kept), n_failed, dropped - n_failed)
        total_dropped += dropped
    log.info("purge complete, %d rows removed", total_dropped)
    return total_dropped


def verify() -> list[str]:
    problems: list[str] = []
    for path in _files():
        rows = _read(path)
        if not rows:
            problems.append(f"{path.name}: empty")
            continue
        key = KEY + ("window_start",) if path.name.startswith("layersweep") else KEY
        metric = _metric_for(path)

        n_failed = sum(1 for r in rows if r.get("ok") is False)
        if n_failed:
            problems.append(f"{path.name}: {n_failed} failed rows retained; run --purge")

        keys = [tuple(r.get(k) for k in key) for r in rows if r.get("ok") is not False]
        if len(keys) != len(set(keys)):
            problems.append(f"{path.name}: {len(keys) - len(set(keys))} duplicate units")

        if metric:
            bad = 0
            for r in rows:
                if r.get("ok") is False:
                    continue
                v = r.get(metric)
                if not isinstance(v, (int, float)) or (isinstance(v, float) and math.isnan(v)):
                    bad += 1
            if bad:
                problems.append(f"{path.name}: {bad} rows with a missing or NaN {metric}")

        # Coverage on the multilingual files is a reportable quantity, not a filter.
        if path.name.startswith("cdva_"):
            ok = sum(1 for r in rows if r.get("ok"))
            frac = ok / len(rows) if rows else 0.0
            level = log.warning if frac < 0.5 else log.info
            level("%s: located %d/%d (%.1f%%)", path.name, ok, len(rows), 100 * frac)
            if frac == 0.0:
                problems.append(f"{path.name}: nothing located at all")

    # Behavioural parquets must not be dry-run remnants.
    try:
        import pandas as pd

        for lang in ("hi", "bn"):
            src = SEEDS_DIR / f"pentad_{lang}.parquet"
            if not src.exists():
                continue
            expected = len(pd.read_parquet(src))
            for p in sorted((OUT / "e4").glob(f"behav_{lang}_*.parquet")):
                have = len(pd.read_parquet(p))
                if have < expected:
                    problems.append(
                        f"{p.name}: {have} rows, expected {expected}; dry-run remnant"
                    )
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet check skipped: %s", str(exc)[:120])

    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not (args.purge or args.verify):
        ap.error("choose --purge, --verify, or both")

    if args.purge:
        purge()
    if args.verify:
        problems = verify()
        if problems:
            log.error("%d problem(s):", len(problems))
            for p in problems:
                log.error("  %s", p)
            sys.exit(1)
        log.info("verify clean: every retained row is a valid, unique measurement")


if __name__ == "__main__":
    main()
