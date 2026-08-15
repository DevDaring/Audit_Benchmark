"""
File: TIST/failure_analysis.py
Purpose: Answer two objections a TIST reviewer will raise about the low MIRAGE-Full rates.

  Objection 1 (review section 8): "the conjunction of five behavioural checks plus a causal
  criterion is simply too strict, which is why the rates are low." That is an empirical
  question. This module decomposes the behavioural verdict into its five components,
  reports where seeds actually fail, and recomputes the pass rate under looser aggregation
  rules (4 of 5, majority) so a reader can see how much of the result is the conjunction.

  Objection 2 (review section 9): "is the result driven entirely by BBQ?" The seeds come
  from three benchmarks with different item forms and scoring semantics. This module
  reports the audit per source and a macro-average across sources, so pooled numbers are
  never the only ones on offer.

Both analyses read saved behavioural and causal outputs. No model inference is run.

Implements / builds on / cites:
  - Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering."
    Findings of ACL 2022.
  - Nangia et al. (2020). "CrowS-Pairs." EMNLP 2020.
  - Nadeem et al. (2021). "StereoSet." ACL 2021.

Usage:
  python TIST/failure_analysis.py

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

log = logging.getLogger("failure_analysis")

TIST = RESULTS_DIR / "tist"
E1 = TIST / "e1"
OUT = TIST / "seed_level"
_OSM = [m["name"] for m in OSM_MODELS]
N_BOOT = 5000

# The five behavioural components, in pentad order.
COMPONENTS = [
    ("correct_a", "(a) surface"),
    ("correct_b", "(b) neutralised"),
    ("stable_c", "(c) substitution"),
    ("correct_d", "(d) context"),
    ("cot_robust", "(e) reasoning"),
]


def _seed_absC() -> pd.DataFrame:
    """Per (seed, model) maximum |C|, which is the quantity the causal gate thresholds."""
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
            d = pd.DataFrame(recs, columns=["seed_id", "absC"])
            g = d.groupby("seed_id")["absC"].max().reset_index()
            g["model_name"] = model
            rows.append(g)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _signals() -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from TIST.e5_api_surrogate import build_signal_table

    behav = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet")
    sig = build_signal_table(behav)
    return sig[sig["model_name"].isin(_OSM)].copy()


# ---------------------------------------------------------------------------
# 1. Where do seeds actually fail?
# ---------------------------------------------------------------------------
def decomposition(tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    sig = _signals()
    absC = _seed_absC()
    m = sig.merge(absC, on=["seed_id", "model_name"], how="inner")
    m["causal_ok"] = m["absC"] <= tau

    rows = []
    for model in _OSM:
        g = m[m["model_name"] == model]
        if g.empty:
            continue
        rec = {"model_name": model, "n_seeds": int(len(g))}
        for col, label in COMPONENTS:
            rec[col] = float(g[col].astype(bool).mean())
        rec["behavioural_all5"] = float(g[[c for c, _ in COMPONENTS]].astype(bool)
                                        .all(axis=1).mean())
        rec["causal_ok"] = float(g["causal_ok"].mean())
        rec["mirage_full"] = float((g[[c for c, _ in COMPONENTS]].astype(bool).all(axis=1)
                                    & g["causal_ok"]).mean())
        # Of the seeds that fail overall, what share fail only behaviourally, only
        # causally, or both? This is what tells a reviewer whether one stage dominates.
        b = g[[c for c, _ in COMPONENTS]].astype(bool).all(axis=1)
        c = g["causal_ok"].astype(bool)
        fail = ~(b & c)
        n_fail = int(fail.sum())
        rec["n_failing"] = n_fail
        if n_fail:
            rec["fail_behavioural_only"] = float(((~b) & c)[fail].mean())
            rec["fail_causal_only"] = float((b & (~c))[fail].mean())
            rec["fail_both"] = float(((~b) & (~c))[fail].mean())
        rows.append(rec)
    comp = pd.DataFrame(rows)

    # Sensitivity to the aggregation rule.
    rng = np.random.default_rng(RANDOM_SEED)
    rules = []
    for model in _OSM:
        g = m[m["model_name"] == model]
        if g.empty:
            continue
        passed = g[[c for c, _ in COMPONENTS]].astype(bool).sum(axis=1)
        c = g["causal_ok"].astype(bool)
        for name, k in (("all 5", 5), ("at least 4", 4), ("majority (3)", 3)):
            b = passed >= k
            full = (b & c).astype(float).values
            draws = full[rng.integers(0, len(full), size=(N_BOOT, len(full)))].mean(axis=1)
            rules.append({
                "model_name": model, "rule": name,
                "behavioural": float(b.mean()), "mirage_full": float(full.mean()),
                "ci_lo": float(np.percentile(draws, 2.5)),
                "ci_hi": float(np.percentile(draws, 97.5)),
            })
    rule_df = pd.DataFrame(rules)

    OUT.mkdir(parents=True, exist_ok=True)
    comp.to_csv(OUT / "component_decomposition.csv", index=False)
    rule_df.to_csv(OUT / "aggregation_rules.csv", index=False)
    return comp, rule_df


# ---------------------------------------------------------------------------
# 2. Is the result driven by one benchmark?
# ---------------------------------------------------------------------------
def by_benchmark(tau: float) -> pd.DataFrame:
    sig = _signals()
    absC = _seed_absC()
    behav = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet",
                            columns=["seed_id", "seed_source"]).drop_duplicates()
    m = (sig.merge(absC, on=["seed_id", "model_name"], how="inner")
            .merge(behav, on="seed_id", how="left"))
    m["causal_ok"] = m["absC"] <= tau
    m["b_all"] = m[[c for c, _ in COMPONENTS]].astype(bool).all(axis=1)
    m["full"] = m["b_all"] & m["causal_ok"]

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in _OSM:
        g = m[m["model_name"] == model]
        if g.empty:
            continue
        per_src = []
        for src, gs in g.groupby("seed_source"):
            v = gs["full"].astype(float).values
            draws = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
            rows.append({
                "model_name": model, "seed_source": src, "n_seeds": int(len(gs)),
                "mirage_b": float(gs["b_all"].mean()),
                "mirage_full": float(v.mean()),
                "ci_lo": float(np.percentile(draws, 2.5)),
                "ci_hi": float(np.percentile(draws, 97.5)),
            })
            per_src.append(v.mean())
        rows.append({
            "model_name": model, "seed_source": "macro-average",
            "n_seeds": int(len(g)),
            "mirage_b": float(g["b_all"].mean()),
            "mirage_full": float(np.mean(per_src)),
            "ci_lo": np.nan, "ci_hi": np.nan,
        })
        rows.append({
            "model_name": model, "seed_source": "pooled",
            "n_seeds": int(len(g)),
            "mirage_b": float(g["b_all"].mean()),
            "mirage_full": float(g["full"].mean()),
            "ci_lo": np.nan, "ci_hi": np.nan,
        })
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "by_benchmark.csv", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stats = E1 / "stats_e1_4_controls.csv"
    tau = float(pd.read_csv(stats)["youden_threshold_absC"].median())
    log.info("using calibrated tau = %.4f", tau)

    comp, rules = decomposition(tau)
    print("\n=== COMPONENT PASS RATES (seed level) ===")
    cols = ["model_name", "n_seeds"] + [c for c, _ in COMPONENTS] + \
           ["behavioural_all5", "causal_ok", "mirage_full"]
    print(comp[cols].round(3).to_string(index=False))

    print("\n=== WHERE FAILING SEEDS FAIL ===")
    print(comp[["model_name", "n_failing", "fail_behavioural_only",
                "fail_causal_only", "fail_both"]].round(3).to_string(index=False))

    print("\n=== SENSITIVITY TO THE AGGREGATION RULE ===")
    print(rules.round(3).to_string(index=False))

    bench = by_benchmark(tau)
    print("\n=== BY SOURCE BENCHMARK ===")
    print(bench.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
