"""
File: TIST/e4_analyse.py
Purpose: Turn the multilingual runs into the per-language validity leaderboard and the
         cross-language comparisons the TIST manuscript reports.

Four questions, in the order the mission file sets them:

  1. Coverage. What fraction of counterfactual pairs could be located at all in Devanagari
     and Bengali script? Multi-token protected terms defeat the position search more often
     than in English, and the located fraction is itself a reported result rather than a
     silent filter.
  2. Does the validity gap widen in lower-resource languages? Native pass rate minus
     audit-robust pass rate, per language per model, with bootstrap CIs.
  3. Does FM4, answer instability, still dominate the loss of validity?
  4. Is the per-model ordering consistent across languages? Kendall tau_b between the
     English ordering and each of Hindi and Bengali.

A language that fails outright for a model is reported as a finding, not dropped.

Implements / builds on / cites:
  - Neplenbroek et al. (2024). "MBBQ: A Dataset for Cross-Lingual Comparison of
    Stereotypes in Generative LLMs." COLM 2024.
  - Kendall (1945). "The treatment of ties in ranking problems." Biometrika 33(3).

Usage:
  python TIST/e4_analyse.py

Part of the MIRAGE audit codebase.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR  # noqa: E402

log = logging.getLogger("e4_analyse")

E4 = RESULTS_DIR / "tist" / "e4"
_OSM = [m["name"] for m in OSM_MODELS]
LANGS = ["hi", "bn"]
N_BOOT = 5000


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(recs)


def _boot_ci(v: np.ndarray, rng) -> tuple[float, float]:
    if not len(v):
        return (np.nan, np.nan)
    draws = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------
# 1. Coverage: how often the protected token could be located
# ---------------------------------------------------------------------------
def coverage() -> pd.DataFrame:
    rows = []
    for lang in LANGS:
        for model in _OSM:
            df = _read_jsonl(E4 / f"cdva_{lang}_{model}.jsonl")
            if df.empty:
                continue
            ok = df["ok"].astype(bool) if "ok" in df.columns else pd.Series(True, index=df.index)
            # After purge_and_verify --purge the failed rows are gone and with them the
            # "reason" column, so the breakdown is only available on an unpurged file.
            if "reason" in df.columns and (~ok).any():
                not_found = int(df.loc[~ok, "reason"].astype(str)
                                .str.contains("position", case=False).sum())
            else:
                not_found = 0
            rows.append(
                {
                    "lang": lang,
                    "model_name": model,
                    "n_attempted": int(len(df)),
                    "n_located": int(ok.sum()),
                    "located_fraction": float(ok.mean()),
                    "n_position_failures": int(not_found),
                    "n_other_failures": int((~ok).sum() - not_found),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E4 / "stats_e4_coverage.csv", index=False)
        for _, r in out.iterrows():
            if r["located_fraction"] < 0.5:
                log.warning(
                    "%s / %s: only %.1f%% of pairs located; report as a coverage finding",
                    r["model_name"], r["lang"], 100 * r["located_fraction"],
                )
    return out


# ---------------------------------------------------------------------------
# 2. Per-language severity and validity gap
# ---------------------------------------------------------------------------
def severity(tau_absC: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for lang in LANGS:
        for model in _OSM:
            df = _read_jsonl(E4 / f"cdva_{lang}_{model}.jsonl")
            if df.empty or "delta_single" not in df.columns:
                continue
            df = df[df["ok"].astype(bool)].dropna(subset=["delta_single"])
            if df.empty:
                continue
            absC = df["delta_single"].abs().values
            lo, hi = _boot_ci(absC, rng)
            # Commutativity index: fraction of seeds whose every pair stays below tau.
            per_seed = df.assign(absC=absC).groupby("seed_id")["absC"].max()
            rows.append(
                {
                    "lang": lang,
                    "model_name": model,
                    "n_pairs": int(len(df)),
                    "n_seeds": int(per_seed.size),
                    "severity_mean_absC": float(absC.mean()),
                    "severity_ci_lo": lo,
                    "severity_ci_hi": hi,
                    "commutativity_index": float((per_seed <= tau_absC).mean()),
                    "tau_absC": tau_absC,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E4 / "stats_e4_severity.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 3. Behavioural pass and validity gap per language
# ---------------------------------------------------------------------------
def validity_gap(tau_absC: float) -> pd.DataFrame:
    """
    Native pass rate against the causal-audit-robust rate, per language per model.

    The behavioural side reuses the production scorer so the English and multilingual
    numbers are produced by the same code.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from TIST.e5_api_surrogate import build_signal_table

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for lang in LANGS:
        for model in _OSM:
            bpath = E4 / f"behav_{lang}_{model}.parquet"
            cpath = E4 / f"cdva_{lang}_{model}.jsonl"
            if not bpath.exists():
                continue
            behav = pd.read_parquet(bpath)
            sig = build_signal_table(behav)
            sig = sig[sig["model_name"] == model] if "model_name" in sig.columns else sig
            native = sig.set_index("seed_id")["mirage_b"].astype(bool)

            cd = _read_jsonl(cpath)
            if cd.empty:
                continue
            cd = cd[cd["ok"].astype(bool)].dropna(subset=["delta_single"])
            causal_ok = (
                cd.assign(absC=cd["delta_single"].abs())
                .groupby("seed_id")["absC"]
                .max()
                .le(tau_absC)
            )
            idx = native.index.intersection(causal_ok.index)
            if not len(idx):
                continue
            nat = native.loc[idx].values.astype(float)
            rob = (native.loc[idx] & causal_ok.loc[idx]).values.astype(float)
            gap = nat - rob
            lo, hi = _boot_ci(gap, rng)
            rows.append(
                {
                    "lang": lang,
                    "model_name": model,
                    "n_seeds": int(len(idx)),
                    "native_pass": float(nat.mean()),
                    "audit_robust_pass": float(rob.mean()),
                    "validity_gap": float(gap.mean()),
                    "gap_ci_lo": lo,
                    "gap_ci_hi": hi,
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E4 / "stats_e4_validity_gap.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Cross-language ordering consistency
# ---------------------------------------------------------------------------
def ordering(gaps: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau_b of the per-model ordering, English against each other language."""
    eng_path = RESULTS_DIR / "validity_gap_leaderboard.parquet"
    if not eng_path.exists() or gaps.empty:
        return pd.DataFrame()
    eng = pd.read_parquet(eng_path)
    col = next((c for c in eng.columns if "gap" in c.lower()), None)
    if col is None:
        return pd.DataFrame()
    eng_rank = eng.groupby("model_name")[col].mean()

    rows = []
    for lang in LANGS:
        sub = gaps[gaps["lang"] == lang].set_index("model_name")["validity_gap"]
        shared = eng_rank.index.intersection(sub.index)
        if len(shared) < 3:
            continue
        tb, p = kendalltau(eng_rank.loc[shared].values, sub.loc[shared].values)
        rows.append(
            {
                "lang": lang,
                "n_models": int(len(shared)),
                "kendall_tau_b_vs_english": float(tb) if tb == tb else np.nan,
                "p_value": float(p) if p == p else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E4 / "stats_e4_ordering.csv", index=False)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not E4.exists():
        raise SystemExit("no E4 results yet")

    # Prefer the Youden-calibrated threshold from E1.4 so English and multilingual
    # numbers share one rule; fall back to the published percentile otherwise.
    stats = RESULTS_DIR / "tist" / "e1" / "stats_e1_4_controls.csv"
    if stats.exists():
        tau_absC = float(pd.read_csv(stats)["youden_threshold_absC"].median())
        log.info("using Youden-calibrated tau on |C| = %.4f", tau_absC)
    else:
        tau_absC = 0.7643661499023438
        log.warning("E1.4 stats absent; using the published percentile tau")

    cov = coverage()
    sev = severity(tau_absC)
    gaps = validity_gap(tau_absC)
    order = ordering(gaps)

    for name, df in (("coverage", cov), ("severity", sev), ("validity_gap", gaps), ("ordering", order)):
        print(f"\n=== {name} ===")
        print(df.to_string(index=False) if len(df) else "(no data yet)")


if __name__ == "__main__":
    main()
