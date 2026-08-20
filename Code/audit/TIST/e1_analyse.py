"""
File: TIST/e1_analyse.py
Purpose: Turn the E1 battery JSONL into the statistics and tables the TIST manuscript
         reports. Runs on CPU, consumes only files under results/tist/.

One function per sub-experiment, each emitting a CSV whose cells are the numbers that
appear in the paper. Nothing is rounded or reformatted for presentation here; the LaTeX
table writer does that, so every printed cell traces to one CSV row.

  e1_1  Placebo separation. Paired Wilcoxon of real |C| against each placebo, per model,
        with rank-biserial correlation and Holm correction across the three comparisons.
  e1_2  Layer localisation. Mean |delta| per layer window per model, normalised within
        model so the profiles are comparable, plus the peak window.
  e1_3  Direction robustness. Agreement of the pass/fail label between the denoising and
        noising directions, Cohen's kappa, and the signed correlation of the magnitudes.
  e1_4  Criterion validity. ROC AUC of |C| separating positive from negative synthetic
        controls, per model, with a bootstrap CI, plus the Youden-optimal threshold that
        E2b adopts.
  e1_5  Convergent validity. Spearman correlation of the commutator with the Vig et al.
        natural indirect effect on the gender subset.
  e1_6  Metric robustness. Spearman correlation of per-seed scores across the
        single-logit, logit-difference and KL metrics.

Implements / builds on / cites:
  - Wilcoxon (1945). "Individual comparisons by ranking methods." Biometrics 1(6):80-83.
  - Holm (1979). "A simple sequentially rejective multiple test procedure."
    Scandinavian Journal of Statistics 6(2):65-70.
  - Kerby (2014). "The simple difference formula: an approach to teaching nonparametric
    correlation." Comprehensive Psychology 3 -- rank-biserial effect size.
  - Youden (1950). "Index for rating diagnostic tests." Cancer 3(1):32-35.
  - Vig et al. (2020). "Investigating Gender Bias in Language Models Using Causal
    Mediation Analysis." NeurIPS 2020.

Usage:
  python TIST/e1_analyse.py

Part of the MIRAGE audit codebase.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR  # noqa: E402

log = logging.getLogger("e1_analyse")

E1 = RESULTS_DIR / "tist" / "e1"
_OSM = [m["name"] for m in OSM_MODELS]
N_BOOT = 5000


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    df = pd.DataFrame(recs)
    return df[df.get("ok", True) == True] if "ok" in df.columns else df  # noqa: E712


def _holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m, adj = len(pvals), [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Kerby's simple difference formula for a paired comparison."""
    d = x - y
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = pd.Series(np.abs(d)).rank().values
    total = ranks.sum()
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / total)


# ---------------------------------------------------------------------------
# E1.1 placebo separation
# ---------------------------------------------------------------------------
def e1_1() -> pd.DataFrame:
    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"battery_{model}.jsonl")
        if df.empty:
            continue
        real = df["real_single"].abs()
        pvals, staged = [], []
        for label in ("content", "function", "shuffled"):
            col = f"placebo_{label}_single"
            if col not in df.columns:
                continue
            sub = df[[c for c in ("real_single", col)]].dropna()
            if len(sub) < 10:
                continue
            a, b = sub["real_single"].abs().values, sub[col].abs().values
            try:
                stat, p = wilcoxon(a, b, alternative="greater")
            except ValueError:
                stat, p = np.nan, 1.0
            pvals.append(float(p))
            staged.append(
                {
                    "model_name": model,
                    "placebo": label,
                    "n_pairs": int(len(sub)),
                    "mean_absC_real": float(a.mean()),
                    "mean_absC_placebo": float(b.mean()),
                    "median_absC_real": float(np.median(a)),
                    "median_absC_placebo": float(np.median(b)),
                    "ratio_real_over_placebo": float(a.mean() / b.mean()) if b.mean() else np.nan,
                    "wilcoxon_stat": float(stat) if stat == stat else np.nan,
                    "p_raw": float(p),
                    "rank_biserial": _rank_biserial(a, b),
                }
            )
        for rec, padj in zip(staged, _holm(pvals)):
            rec["p_holm"] = padj
            rec["significant_0.05"] = bool(padj < 0.05)
            rows.append(rec)
        log.info("e1.1 %s: %d pairs, mean real |C| %.4f", model, len(df), float(real.mean()))
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_1_placebo.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E1.2 layer localisation
# ---------------------------------------------------------------------------
def e1_2() -> pd.DataFrame:
    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"layersweep_{model}.jsonl")
        if df.empty:
            continue
        df["absd"] = df["delta_single"].abs()
        g = df.groupby("window_start").agg(
            mean_absC=("absd", "mean"),
            median_absC=("absd", "median"),
            n=("absd", "size"),
            n_layers=("n_layers", "first"),
        ).reset_index()
        peak = g["mean_absC"].max()
        g["normalised"] = g["mean_absC"] / peak if peak else np.nan
        g["depth_frac"] = g["window_start"] / g["n_layers"]
        g["model_name"] = model
        g["is_peak"] = g["mean_absC"] == peak
        rows.append(g)
        pk = g.loc[g["mean_absC"].idxmax()]
        log.info(
            "e1.2 %s: peak at layers %d+ (depth %.2f), mean |C| %.4f",
            model, int(pk["window_start"]), float(pk["depth_frac"]), float(pk["mean_absC"]),
        )
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(out):
        out.to_csv(E1 / "stats_e1_2_layers.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E1.3 direction robustness
# ---------------------------------------------------------------------------
def e1_3(tau_absC: float) -> pd.DataFrame:
    from TIST.e3_annotation import cohen_kappa

    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"battery_{model}.jsonl")
        if df.empty or "reverse_single" not in df.columns:
            continue
        sub = df[["real_single", "reverse_single"]].dropna()
        fwd = (sub["real_single"].abs() > tau_absC).astype(int).values
        rev = (sub["reverse_single"].abs() > tau_absC).astype(int).values
        rho, p = spearmanr(sub["real_single"].abs(), sub["reverse_single"].abs())
        rows.append(
            {
                "model_name": model,
                "n_pairs": int(len(sub)),
                "tau_absC": tau_absC,
                "label_agreement": float((fwd == rev).mean()),
                "cohen_kappa_direction": cohen_kappa(fwd, rev),
                "spearman_magnitude": float(rho),
                "spearman_p": float(p),
                "mean_absC_denoising": float(sub["real_single"].abs().mean()),
                "mean_absC_noising": float(sub["reverse_single"].abs().mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_3_direction.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E1.4 criterion validity
# ---------------------------------------------------------------------------
def e1_4() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"controls_{model}.jsonl")
        if df.empty:
            continue
        df = df.dropna(subset=["delta_single", "control_type"])
        # Aggregate to the seed, matching how the audit scores a seed.
        seed = (
            df.assign(absd=df["delta_single"].abs())
            .groupby(["seed_id", "control_type"])["absd"]
            .mean()
            .reset_index()
        )
        y = (seed["control_type"] == "positive").astype(int).values
        s = seed["absd"].values
        if len(set(y.tolist())) < 2:
            continue
        auc = float(roc_auc_score(y, s))

        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            if len(set(y[idx].tolist())) < 2:
                continue
            boots.append(roc_auc_score(y[idx], s[idx]))
        lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)

        fpr, tpr, thr = roc_curve(y, s)
        j = np.argmax(tpr - fpr)
        rows.append(
            {
                "model_name": model,
                "n_positive_seeds": int(y.sum()),
                "n_negative_seeds": int((1 - y).sum()),
                "auc": auc,
                "auc_ci_lo": lo,
                "auc_ci_hi": hi,
                "meets_0.9_target": bool(auc >= 0.9),
                "youden_threshold_absC": float(thr[j]),
                "youden_j": float(tpr[j] - fpr[j]),
                "sensitivity_at_youden": float(tpr[j]),
                "specificity_at_youden": float(1 - fpr[j]),
                "mean_absC_positive": float(s[y == 1].mean()),
                "mean_absC_negative": float(s[y == 0].mean()),
            }
        )
        log.info("e1.4 %s: AUC %.3f [%.3f, %.3f], Youden tau %.3f", model, auc, lo, hi, float(thr[j]))
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_4_controls.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E1.5 convergent validity
# ---------------------------------------------------------------------------
def e1_5() -> pd.DataFrame:
    rows = []
    for model in _OSM:
        med = _read_jsonl(E1 / f"mediation_{model}.jsonl")
        bat = _read_jsonl(E1 / f"battery_{model}.jsonl")
        if med.empty or bat.empty:
            continue
        key = ["seed_id", "sub_a", "sub_b"]
        m = med.merge(bat[key + ["real_single"]], on=key, how="inner")
        if len(m) < 20:
            continue
        rho, p = spearmanr(m["real_single"].abs(), m["indirect_effect"].abs())
        rho_signed, p_signed = spearmanr(m["real_single"], m["indirect_effect"])
        rows.append(
            {
                "model_name": model,
                "n_pairs": int(len(m)),
                "spearman_absolute": float(rho),
                "p_absolute": float(p),
                "spearman_signed": float(rho_signed),
                "p_signed": float(p_signed),
                "mean_indirect": float(m["indirect_effect"].mean()),
                "mean_direct": float(m["direct_effect"].mean()),
                "mean_total": float(m["total_effect"].mean()),
                "indirect_share": float(
                    np.abs(m["indirect_effect"]).mean() / np.abs(m["total_effect"]).mean()
                ) if np.abs(m["total_effect"]).mean() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_5_mediation.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# E1.6 metric robustness
# ---------------------------------------------------------------------------
def e1_6() -> pd.DataFrame:
    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"battery_{model}.jsonl")
        if df.empty:
            continue
        cols = [c for c in ("real_single", "real_logitdiff", "real_kl") if c in df.columns]
        if len(cols) < 2:
            continue
        seed = df.groupby("seed_id")[cols].apply(lambda g: g.abs().mean())
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                sub = seed[[cols[i], cols[j]]].dropna()
                if len(sub) < 20:
                    continue
                rho, p = spearmanr(sub[cols[i]], sub[cols[j]])
                rows.append(
                    {
                        "model_name": model,
                        "metric_a": cols[i].replace("real_", ""),
                        "metric_b": cols[j].replace("real_", ""),
                        "n_seeds": int(len(sub)),
                        "spearman": float(rho),
                        "p_value": float(p),
                    }
                )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_6_metrics.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Cross-stack consistency (needed because the patching stack moved)
# ---------------------------------------------------------------------------
def consistency() -> pd.DataFrame:
    """
    Compare the battery's real patch against the stored production commutator.

    The production CDVA ran on torch 2.5.1 + cu124 with flash-attention; the TIST battery
    runs on torch 2.13.0 + cu130 with sdpa, because the pinned torch did not survive
    dependency resolution in the container. E1's internal comparisons are unaffected,
    since every condition runs on the same stack, but E1.4 and E4 are read against stored
    English values and that comparison assumes the two stacks agree.

    No extra inference is needed to check it. `real_single` is computed exactly as
    GPU_CPU/utils_attention.patch_activation computes `delta_logit`: every layer patched
    at the protected position, read as the change in the bias-answer logit, with the same
    bias-answer rule. Pairs present in both runs can therefore be correlated directly.

    Only clean pairs overlap: production ran on the raw pentad, the battery on the pentad
    with the 297 degenerate slot-c variants removed, so the degenerate pairs are absent
    here by construction.

    A high correlation with a small mean absolute difference licenses the cross-run
    comparison. A low one means the multilingual and control results must be read against
    a re-run English baseline instead, and the paper must say so.
    """
    stored_path = RESULTS_DIR / "cdva_results.parquet"
    if not stored_path.exists():
        return pd.DataFrame()
    stored = pd.read_parquet(stored_path)
    stored = stored[stored["success_flag"] == True]  # noqa: E712

    rows = []
    for model in _OSM:
        df = _read_jsonl(E1 / f"battery_{model}.jsonl")
        if df.empty or "real_single" not in df.columns:
            continue
        s = stored[stored["model_name"] == model][
            ["seed_id", "pair_A_subvariant", "pair_B_subvariant", "delta_logit"]
        ].rename(columns={"pair_A_subvariant": "sub_a", "pair_B_subvariant": "sub_b"})
        m = df[["seed_id", "sub_a", "sub_b", "real_single"]].merge(
            s, on=["seed_id", "sub_a", "sub_b"], how="inner"
        ).dropna()
        if len(m) < 30:
            continue
        diff = (m["real_single"] - m["delta_logit"]).abs()
        pear = float(np.corrcoef(m["real_single"], m["delta_logit"])[0, 1])
        rho, _ = spearmanr(m["real_single"], m["delta_logit"])
        rows.append(
            {
                "model_name": model,
                "n_overlapping_pairs": int(len(m)),
                "pearson": pear,
                "spearman": float(rho),
                "mean_abs_diff": float(diff.mean()),
                "median_abs_diff": float(diff.median()),
                "p95_abs_diff": float(diff.quantile(0.95)),
                "mean_absC_stored": float(m["delta_logit"].abs().mean()),
                "mean_absC_rerun": float(m["real_single"].abs().mean()),
                # A difference worth caring about is one large enough to move a pair
                # across the threshold, so it is expressed relative to the signal.
                "mean_abs_diff_over_mean_absC": float(
                    diff.mean() / m["delta_logit"].abs().mean()
                ) if m["delta_logit"].abs().mean() else np.nan,
                "cross_stack_comparable": bool(pear >= 0.99 and diff.mean() < 0.05),
            }
        )
        log.info(
            "consistency %s: n=%d pearson=%.4f mean|diff|=%.4f",
            model, len(m), pear, float(diff.mean()),
        )
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(E1 / "stats_e1_7_consistency.csv", index=False)
        if not out["cross_stack_comparable"].all():
            log.warning(
                "cross-stack agreement is weaker than the 0.99 / 0.05 bar on at least one "
                "model; E1.4 and E4 must be read against a re-run English baseline and the "
                "manuscript must state this"
            )
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not E1.exists():
        raise SystemExit("no E1 results yet; run TIST/run_tist_gpu.py first")

    # The direction test needs a threshold on |C|. Prefer the Youden threshold the
    # controls give, since that is the calibrated one E2b adopts; fall back to the
    # published percentile rule when the controls have not run.
    controls = e1_4()
    if len(controls):
        tau_absC = float(controls["youden_threshold_absC"].median())
        log.info("using Youden-calibrated tau on |C| = %.4f", tau_absC)
    else:
        tau_absC = 0.7643661499023438
        log.warning("controls absent; falling back to the published percentile tau")

    results = {
        "e1_1_placebo": e1_1(),
        "e1_2_layers": e1_2(),
        "e1_3_direction": e1_3(tau_absC),
        "e1_4_controls": controls,
        "e1_5_mediation": e1_5(),
        "e1_6_metrics": e1_6(),
        "e1_7_consistency": consistency(),
    }

    summary = {k: (len(v) if isinstance(v, pd.DataFrame) else 0) for k, v in results.items()}
    summary["tau_absC_used"] = tau_absC
    (E1 / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for name, df in results.items():
        print(f"\n=== {name} ===")
        print(df.to_string(index=False) if len(df) else "(no data yet)")


if __name__ == "__main__":
    main()
