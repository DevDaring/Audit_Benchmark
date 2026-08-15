"""
File: TIST/e2b_calibrated_tau.py
Purpose: E2b -- replace the percentile threshold with one calibrated against ground truth,
         and recompute every MIRAGE-Full number under it.

Reviewer 2 objected that tau as the 75th percentile of |C| is heuristic. E2a showed the
objection understated the problem: the published tau was computed in logit units and then
compared against a [0,1] invariance score, so the effective cut sat at the 85.4th
percentile rather than the 75th, and the middle two models swap rank around the 80th.

A percentile is a distributional convention, not a decision rule. This module replaces it
with a threshold chosen against items whose correct causal verdict is known:
TIST/e1_4_build_controls builds 100 positive controls whose answer must depend on the
protected attribute and 100 negatives where it provably cannot, and E1.4 measures how well
|C| separates them. The Youden-optimal cut on that ROC is the threshold that maximises
sensitivity plus specificity minus one, so it is calibrated to the audit's own criterion
rather than to the shape of its score distribution.

Reported both ways, because the change moves headline numbers and hiding that would be
worse than the original error.

Implements / builds on / cites:
  - Youden (1950). "Index for rating diagnostic tests." Cancer 3(1):32-35.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.

Usage:
  python TIST/e2b_calibrated_tau.py

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR  # noqa: E402

log = logging.getLogger("e2b")

OUT = RESULTS_DIR / "tist" / "e2"
E1 = RESULTS_DIR / "tist" / "e1"
PUBLISHED_TAU_SCORE = 0.7643661499023438   # as shipped, compared against the [0,1] score
CDVA_SCALE = 5.0
N_BOOT = 5000
_OSM = [m["name"] for m in OSM_MODELS]


def _seed_absC() -> pd.DataFrame:
    """Per (seed, model) mean |C|, from the re-run battery so the stack is consistent."""
    rows = []
    for model in _OSM:
        p = E1 / f"battery_{model}.jsonl"
        if not p.exists():
            continue
        recs = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") and isinstance(r.get("real_single"), (int, float)):
                recs.append((r["seed_id"], abs(r["real_single"])))
        if recs:
            df = pd.DataFrame(recs, columns=["seed_id", "absC"])
            g = df.groupby("seed_id")["absC"].mean().reset_index()
            g["model_name"] = model
            rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _boot_ci(flags: np.ndarray, rng) -> tuple[float, float]:
    if not len(flags):
        return (np.nan, np.nan)
    draws = flags[rng.integers(0, len(flags), size=(N_BOOT, len(flags)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    ctrl_path = E1 / "stats_e1_4_controls.csv"
    if not ctrl_path.exists():
        raise SystemExit("run TIST/e1_analyse.py first to produce the E1.4 controls table")
    ctrl = pd.read_csv(ctrl_path)

    # One threshold for the whole audit, the median of the per-model Youden cuts. A
    # per-model threshold would make the leaderboard incomparable across models, which is
    # the thing the leaderboard exists to do.
    tau_absC = float(ctrl["youden_threshold_absC"].median())
    log.info("calibrated tau on |C| = %.4f (median of per-model Youden cuts)", tau_absC)

    seeds = _seed_absC()
    if seeds.empty:
        raise SystemExit("no battery results found")

    scored = pd.read_parquet(RESULTS_DIR / "scored_results.parquet")
    scored = scored[scored["model_name"].isin(_OSM)]
    merged = scored.merge(seeds, on=["seed_id", "model_name"], how="inner")
    log.info("scoring %d seed x model rows", len(merged))

    published_absC_cut = CDVA_SCALE * (1.0 - PUBLISHED_TAU_SCORE)
    rng = np.random.default_rng(RANDOM_SEED)

    rows = []
    for model in _OSM:
        sub = merged[merged["model_name"] == model]
        if sub.empty:
            continue
        b = sub["mirage_b_pass"].astype(bool).values
        cal = (b & (sub["absC"] <= tau_absC).values).astype(float)
        pub = (b & (sub["absC"] <= published_absC_cut).values).astype(float)
        lo, hi = _boot_ci(cal, rng)
        rows.append({
            "model_name": model,
            "n_seeds": int(len(sub)),
            "mirage_b": float(b.mean()),
            "mirage_full_published_rule": float(pub.mean()),
            "mirage_full_calibrated": float(cal.mean()),
            "calibrated_ci_lo": lo,
            "calibrated_ci_hi": hi,
            "change": float(cal.mean() - pub.mean()),
            "tau_absC_calibrated": tau_absC,
            "tau_absC_published_effective": published_absC_cut,
        })

    out = pd.DataFrame(rows).sort_values("mirage_full_calibrated", ascending=False)
    out.to_csv(OUT / "stats_e2b_calibrated.csv", index=False)

    summary = {
        "tau_absC_calibrated": tau_absC,
        "tau_absC_published_effective": published_absC_cut,
        "per_model_youden": ctrl.set_index("model_name")["youden_threshold_absC"].to_dict(),
        "ordering_calibrated": out["model_name"].tolist(),
        "note": (
            "The calibrated threshold comes from the E1.4 ground-truth controls, not from a "
            "percentile of the score distribution. A pair counts as causally invariant when "
            "|C| <= tau."
        ),
    }
    (OUT / "e2b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(out.to_string(index=False))
    print()
    print("ordering under the calibrated rule:", " > ".join(out["model_name"]))


if __name__ == "__main__":
    main()
