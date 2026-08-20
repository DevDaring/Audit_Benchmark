"""
File: TIST/tau_validation.py
Purpose: Test whether the threshold and the per-seed causal rule are doing the work a
         reader would suspect them of doing.

Two questions, both answerable from saved outputs.

  1. THRESHOLD (review section 7). tau was chosen to maximise Youden's J on the 200
     synthetic controls, and then applied to the same controls when the AUC was reported.
     That is in-sample calibration. This module splits the controls in half by seed,
     calibrates on one half and evaluates on the other, repeats over many splits, and
     reports the held-out gap. It also sweeps tau across a wide range to show which
     conclusions survive and which do not.

  2. PER-SEED RULE. A seed passes the causal stage only when EVERY one of its pairs stays
     below tau, so the statistic is a maximum over roughly sixteen pairs. A maximum grows
     with the number of draws, which means the rule is conservative by construction and
     the pass rate is not comparable across seeds with different pair counts. This module
     recomputes the causal pass rate under three rules -- max, mean, and the fraction of
     pairs below tau -- so the paper can state how much the choice matters.

No model inference is run.

Implements / builds on / cites:
  - Youden (1950). "Index for rating diagnostic tests." Cancer 3(1):32-35.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.

Usage:
  python TIST/tau_validation.py

Part of the MIRAGE audit codebase.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR  # noqa: E402

log = logging.getLogger("tau_validation")

TIST = RESULTS_DIR / "tist"
E1 = TIST / "e1"
OUT = TIST / "seed_level"
_OSM = [m["name"] for m in OSM_MODELS]
N_SPLITS = 200


def _controls(model: str) -> pd.DataFrame:
    """Per control seed: the |C| the audit scores it on, and its known verdict."""
    p = E1 / f"controls_{model}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    recs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The control runs write delta_single; the main battery writes real_single.
        v = r.get("delta_single", r.get("real_single"))
        if r.get("ok") and isinstance(v, (int, float)):
            recs.append({"seed_id": r["seed_id"], "absC": abs(v),
                         "control_type": r.get("control_type")})
    d = pd.DataFrame(recs)
    if not d.empty and d["control_type"].isna().all():
        # Fall back to the seed naming convention if the run did not carry the label.
        d["control_type"] = np.where(d["seed_id"].str.contains("pos"),
                                     "positive", "negative")
    return d


def _youden(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_validation() -> pd.DataFrame:
    """Split-half calibration: choose tau on one half of the control seeds, score the other."""
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in _OSM:
        d = _controls(model)
        if d.empty:
            continue
        per_seed = d.groupby(["seed_id", "control_type"])["absC"].mean().reset_index()
        y = (per_seed["control_type"] == "positive").astype(int).values
        s = per_seed["absC"].values
        if len(np.unique(y)) < 2:
            continue

        in_sample_tau = _youden(y, s)
        in_sample_auc = float(roc_auc_score(y, s))

        taus, accs = [], []
        n = len(y)
        for _ in range(N_SPLITS):
            idx = rng.permutation(n)
            a, b = idx[: n // 2], idx[n // 2:]
            if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
                continue
            t = _youden(y[a], s[a])
            taus.append(t)
            accs.append(float(((s[b] >= t).astype(int) == y[b]).mean()))
        if not accs:
            continue
        in_acc = float(((s >= in_sample_tau).astype(int) == y).mean())
        rows.append({
            "model_name": model, "n_control_seeds": n,
            "auc_in_sample": in_sample_auc,
            "tau_in_sample": in_sample_tau,
            "tau_heldout_mean": float(np.mean(taus)),
            "tau_heldout_sd": float(np.std(taus)),
            "accuracy_in_sample": in_acc,
            "accuracy_heldout_mean": float(np.mean(accs)),
            "optimism": in_acc - float(np.mean(accs)),
        })
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "threshold_validation.csv", index=False)
    return out


def _seed_pairs(model: str) -> pd.DataFrame:
    p = E1 / f"battery_{model}.jsonl"
    if not p.exists():
        return pd.DataFrame()
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
    return pd.DataFrame(recs, columns=["seed_id", "absC"])


def rule_sensitivity(tau: float) -> pd.DataFrame:
    """Causal pass rate under three ways of reducing a seed's pairs to one verdict."""
    rows = []
    for model in _OSM:
        d = _seed_pairs(model)
        if d.empty:
            continue
        g = d.groupby("seed_id")["absC"]
        n_pairs = g.size()
        rows.append({
            "model_name": model,
            "n_seeds": int(n_pairs.size),
            "median_pairs_per_seed": float(n_pairs.median()),
            "rule_max": float((g.max() <= tau).mean()),
            "rule_mean": float((g.mean() <= tau).mean()),
            "rule_frac80": float((d.assign(ok=d["absC"] <= tau)
                                  .groupby("seed_id")["ok"].mean() >= 0.8).mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "causal_rule_sensitivity.csv", index=False)
    return out


def tau_sweep() -> pd.DataFrame:
    """How the causal pass rate moves as tau varies, under the shipped max rule."""
    grid = [0.125, 0.25, 0.375, 0.456, 0.5, 0.75, 1.0, 1.5, 2.0]
    rows = []
    for model in _OSM:
        d = _seed_pairs(model)
        if d.empty:
            continue
        g = d.groupby("seed_id")["absC"].max()
        for t in grid:
            rows.append({"model_name": model, "tau": t,
                         "causal_pass": float((g <= t).mean())})
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "tau_sweep.csv", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tau = float(pd.read_csv(E1 / "stats_e1_4_controls.csv")["youden_threshold_absC"].median())
    log.info("shipped tau = %.4f", tau)

    tv = threshold_validation()
    print("\n=== THRESHOLD: IN-SAMPLE vs HELD-OUT ===")
    print(tv.round(4).to_string(index=False) if len(tv) else "(no control outputs)")

    rs = rule_sensitivity(tau)
    print("\n=== PER-SEED CAUSAL RULE ===")
    print(rs.round(4).to_string(index=False))

    sweep = tau_sweep()
    print("\n=== TAU SWEEP (max rule) ===")
    print(sweep.pivot(index="tau", columns="model_name", values="causal_pass")
          .round(3).to_string())


if __name__ == "__main__":
    main()
