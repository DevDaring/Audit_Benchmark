"""
File: TIST/e2_tau_sensitivity.py
Purpose: E2 -- threshold (tau) calibration and sensitivity, answering Reviewer 2
         point 2 of the COMJNL rejection ("the 75th percentile rule is heuristic;
         the absolute pass rates are sensitive to it").

Three things happen here, in order.

1. SCALE AUDIT. The production pipeline carries a unit mismatch. run_cpu_postprocess.py
   computes tau as the 75th percentile of |delta_logit| (logit units), whereas
   CPU_Only/scoring.compute_mirage_full compares it against cdva_seed_score, which is
   1 - min(|delta_logit| / 5, 1), a [0,1] invariance score. The comparison is only
   well-formed by coincidence, because 0.7644 logits happens to fall inside [0,1].
   GPU_CPU/cdva_calibration.py searches candidate taus on np.linspace(0.1, 0.9, 17),
   which confirms tau was designed as a score threshold. This module reports the size
   of the discrepancy rather than hiding it.

2. SCALE-CONSISTENT SWEEP. A percentile p on |C| maps to the score threshold
   tau(p) = 1 - percentile(|C|, p) / CDVA_GLOBAL_SCALE. MIRAGE-Full is recomputed for
   p in [10, 90] with bootstrap CIs over seeds.

3. ORDERING STABILITY. Kendall tau_b between the model ordering at each p and the
   ordering under the published rule, which is the claim the rejected manuscript made
   in its Section 6.8 robustness check.

MIRAGE-B is independent of tau, so it is read from the production scored_results
parquet rather than recomputed. This keeps the sweep exactly consistent with
CPU_Only/scoring.py while staying cheap.

Implements / builds on / cites:
  - Youden (1950). "Index for rating diagnostic tests." Cancer 3(1):32-35.
    -- criterion used by e2b once the E1.4 synthetic controls exist.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap. Chapman & Hall.
  - Kendall (1945). "The treatment of ties in ranking problems." Biometrika 33(3).

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, RANDOM_SEED  # noqa: E402


def _read_global_scale() -> float:
    """
    Read CDVA_GLOBAL_SCALE from its definition in GPU_CPU/cdva_patching.py.

    Importing that module pulls in torch, which is absent on the analysis host.
    Parsing the assignment keeps a single source of truth without the dependency,
    and fails loudly if the constant is renamed or removed.
    """
    src = (Path(__file__).resolve().parents[1] / "GPU_CPU" / "cdva_patching.py").read_text(
        encoding="utf-8"
    )
    for line in src.splitlines():
        if line.startswith("CDVA_GLOBAL_SCALE"):
            return float(line.split("=", 1)[1].split("#")[0].strip())
    raise RuntimeError("CDVA_GLOBAL_SCALE not found in GPU_CPU/cdva_patching.py")


CDVA_GLOBAL_SCALE: float = _read_global_scale()

log = logging.getLogger("e2_tau")

OUT = RESULTS_DIR / "tist" / "e2"
PUBLISHED_TAU = 0.7643661499023438      # value in results/tau_calibration.json
N_BOOT = 5000
PERCENTILES = list(range(10, 95, 5))
_OSM = [m["name"] for m in OSM_MODELS]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-seed-model frame with mirage_b_pass and cdva score, raw cdva)."""
    cdva = pd.read_parquet(RESULTS_DIR / "cdva_results.parquet")
    cdva = cdva[cdva["success_flag"] == True]  # noqa: E712
    if "position_fallback_used" in cdva.columns:
        cdva = cdva[~cdva["position_fallback_used"].astype(bool)]

    scored = pd.read_parquet(RESULTS_DIR / "scored_results.parquet")
    scored = scored[scored["model_name"].isin(_OSM)].copy()

    # compute_mirage_full aggregates the pair scores by mean over successful pairs.
    seed_score = (
        cdva.groupby(["seed_id", "model_name"])["cdva_pair_score"]
        .mean()
        .rename("cdva_seed_score")
        .reset_index()
    )
    frame = scored.merge(seed_score, on=["seed_id", "model_name"], how="left")
    return frame, cdva


# ---------------------------------------------------------------------------
# 1. Scale audit
# ---------------------------------------------------------------------------
def scale_audit(frame: pd.DataFrame, cdva: pd.DataFrame) -> dict:
    """Quantify the unit mismatch and its effect on the published pass rates."""
    absC = cdva["delta_logit"].abs().values

    # What |C| cut does the published tau actually impose once it is compared
    # against a score of the form 1 - |C| / scale?
    implied_cut = CDVA_GLOBAL_SCALE * (1.0 - PUBLISHED_TAU)
    implied_pct = float((absC < implied_cut).mean() * 100.0)

    # The scale-consistent threshold for the SAME 75th percentile of |C|.
    corrected_tau = 1.0 - float(np.percentile(absC, 75)) / CDVA_GLOBAL_SCALE

    published = pass_rates(frame, PUBLISHED_TAU)
    corrected = pass_rates(frame, corrected_tau)

    # Integrity check: the published tau must reproduce the stored labels exactly.
    stored = frame["mirage_full_pass"].astype("boolean")
    recomputed = frame["mirage_b_pass"].astype(bool) & (
        frame["cdva_seed_score"] > PUBLISHED_TAU
    )
    reproduces = bool((stored.fillna(False) == recomputed).all())

    return {
        "published_tau": PUBLISHED_TAU,
        "published_tau_units": "75th percentile of |delta_logit|, in logit units",
        "applied_against": "cdva_seed_score = 1 - min(|delta_logit|/%.1f, 1)"
        % CDVA_GLOBAL_SCALE,
        "implied_absC_cut_logits": implied_cut,
        "implied_absC_percentile": implied_pct,
        "nominal_absC_percentile": 75.0,
        "corrected_tau_same_percentile": corrected_tau,
        "reproduces_stored_labels": reproduces,
        "mirage_full_published": published,
        "mirage_full_corrected": corrected,
        "delta_vs_published": {k: corrected[k] - published[k] for k in published},
    }


# ---------------------------------------------------------------------------
# 2. Sweep
# ---------------------------------------------------------------------------
def pass_rates(frame: pd.DataFrame, tau: float) -> dict[str, float]:
    ok = frame["mirage_b_pass"].astype(bool) & (frame["cdva_seed_score"] > tau)
    return {m: float(v) for m, v in ok.groupby(frame["model_name"]).mean().items()}


def _boot_ci(flags: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(flags)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    draws = flags[idx].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sweep(frame: pd.DataFrame, cdva: pd.DataFrame) -> pd.DataFrame:
    absC = cdva["delta_logit"].abs().values
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for p in PERCENTILES:
        cut = float(np.percentile(absC, p))
        tau = 1.0 - cut / CDVA_GLOBAL_SCALE
        for model in _OSM:
            sub = frame[frame["model_name"] == model]
            flags = (
                sub["mirage_b_pass"].astype(bool) & (sub["cdva_seed_score"] > tau)
            ).values.astype(float)
            lo, hi = _boot_ci(flags, rng)
            rows.append(
                {
                    "percentile": p,
                    "absC_cut_logits": cut,
                    "tau_score_scale": tau,
                    "model_name": model,
                    "n_seeds": int(len(flags)),
                    "mirage_full_rate": float(flags.mean()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Ordering stability
# ---------------------------------------------------------------------------
def ordering_stability(sweep_df: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Kendall tau_b of the model ordering at each percentile against the published rule."""
    ref = pd.Series(pass_rates(frame, PUBLISHED_TAU))[_OSM].values
    rows = []
    for p, g in sweep_df.groupby("percentile"):
        cur = g.set_index("model_name").loc[_OSM, "mirage_full_rate"].values
        tb, pval = kendalltau(ref, cur)
        rows.append(
            {
                "percentile": int(p),
                "kendall_tau_b_vs_published": float(tb) if tb == tb else np.nan,
                "p_value": float(pval) if pval == pval else np.nan,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    frame, cdva = load_inputs()
    log.info("loaded %d seed x model rows, %d cdva pairs", len(frame), len(cdva))

    audit = scale_audit(frame, cdva)
    (OUT / "scale_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log.info(
        "scale audit: published tau imposes a |C| cut at %.3f logits = pct %.1f (nominal 75)",
        audit["implied_absC_cut_logits"],
        audit["implied_absC_percentile"],
    )

    sw = sweep(frame, cdva)
    sw.to_parquet(OUT / "tau_sweep.parquet", index=False)
    sw.to_csv(OUT / "stats_tau_sensitivity.csv", index=False)

    order = ordering_stability(sw, frame)
    order.to_csv(OUT / "stats_tau_ordering.csv", index=False)

    # Explicit rank table. With four models Kendall tau_b has almost no power, so the
    # readable evidence is which ranks actually move and where.
    piv = sw.pivot(index="percentile", columns="model_name", values="mirage_full_rate")
    piv.insert(0, "absC_cut_logits", sw.groupby("percentile")["absC_cut_logits"].first())
    piv.insert(1, "tau_score_scale", sw.groupby("percentile")["tau_score_scale"].first())
    piv["rank_order"] = piv[_OSM].apply(
        lambda r: " > ".join(r.sort_values(ascending=False).index), axis=1
    )
    piv.to_csv(OUT / "stats_tau_rank_table.csv")

    log.info("wrote %s", OUT)
    print(json.dumps(audit, indent=2))
    print()
    print(order.to_string(index=False))


if __name__ == "__main__":
    main()
