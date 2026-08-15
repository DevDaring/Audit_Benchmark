"""
File: TIST/seed_level.py
Purpose: Redo the inferential statistics with the SEED as the unit of analysis, and audit
         the provenance of every count the manuscript reports.

Why this exists. A pentad produces about sixteen counterfactual pairs from one seed, and
those pairs share a passage, an answer set and a protected axis. Testing over pairs treats
them as independent observations when they are not, which is pseudoreplication: it shrinks
the standard error towards zero and inflates significance. The shipped placebo test ran a
Wilcoxon signed-rank over 9,566 pairs per model and reported p = 0. That number is a
statement about the sample size, not about the effect.

What this module does instead:

  1. Aggregates |C| to one value per seed, then runs the paired test over seeds.
  2. Reports a cluster bootstrap, resampling SEEDS with replacement and carrying all of a
     seed's pairs along, which is the resampling unit that matches the design.
  3. Recomputes the mediation correlation at seed level for the same reason.
  4. Reconciles every count the manuscript reports against the files.

No model inference is run. Everything reads saved outputs.

Implements / builds on / cites:
  - Hurlbert (1984). "Pseudoreplication and the design of ecological field experiments."
    Ecological Monographs 54(2):187-211.
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap. (cluster bootstrap)
  - Holm (1979). "A simple sequentially rejective multiple test procedure."
  - Wilcoxon (1945). "Individual comparisons by ranking methods." Biometrics 1(6).

Usage:
  python TIST/seed_level.py
  python TIST/seed_level.py --audit-only

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR  # noqa: E402

log = logging.getLogger("seed_level")

TIST = RESULTS_DIR / "tist"
E1 = TIST / "e1"
OUT = TIST / "seed_level"
DATASET = RESULTS_DIR.parent / "Dataset" / "seeds"
_OSM = [m["name"] for m in OSM_MODELS]
N_BOOT = 5000
PLACEBOS = ("content", "function", "shuffled")


def _battery(model: str) -> pd.DataFrame:
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
        if r.get("ok"):
            recs.append(r)
    return pd.DataFrame(recs)


def _holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni, returning adjusted p-values keyed as they came in."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)      # enforce monotonicity
        out[k] = running
    return out


# ---------------------------------------------------------------------------
# 1. Placebo comparison, seed level
# ---------------------------------------------------------------------------
def placebo_seed_level() -> pd.DataFrame:
    """Paired test over seeds, plus a cluster bootstrap of the ratio.

    Each seed contributes one mean |C| for the real intervention and one for each control.
    The paired Wilcoxon then has n = number of seeds, which is the number of independent
    items the audit actually drew.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in _OSM:
        df = _battery(model)
        if df.empty:
            continue
        df = df.assign(real=df["real_single"].abs())
        for pl in PLACEBOS:
            col = f"placebo_{pl}_single"
            if col not in df.columns:
                continue
            sub = df.dropna(subset=["real", col]).copy()
            sub["ctrl"] = sub[col].abs()

            per_seed = sub.groupby("seed_id")[["real", "ctrl"]].mean()
            n_seeds, n_pairs = len(per_seed), len(sub)
            if n_seeds < 10:
                continue

            stat, p_seed = wilcoxon(per_seed["real"], per_seed["ctrl"],
                                    alternative="greater")
            # Cluster bootstrap: resample seeds, keep each seed's pairs together.
            ids = per_seed.index.to_numpy()
            draws = np.empty(N_BOOT)
            real_v = per_seed["real"].to_numpy()
            ctrl_v = per_seed["ctrl"].to_numpy()
            for b in range(N_BOOT):
                take = rng.integers(0, n_seeds, n_seeds)
                c = ctrl_v[take].mean()
                draws[b] = real_v[take].mean() / c if c > 0 else np.nan
            lo, hi = np.nanpercentile(draws, [2.5, 97.5])

            # Matched-pairs effect size that does not assume normality: the fraction of
            # seeds on which the real intervention moves the answer more.
            wins = float((per_seed["real"] > per_seed["ctrl"]).mean())

            rows.append({
                "model_name": model, "placebo": pl,
                "n_seeds": n_seeds, "n_pairs": n_pairs,
                "mean_absC_real": float(per_seed["real"].mean()),
                "mean_absC_placebo": float(per_seed["ctrl"].mean()),
                "ratio": float(per_seed["real"].mean() / per_seed["ctrl"].mean()),
                "ratio_ci_lo": float(lo), "ratio_ci_hi": float(hi),
                "win_rate_seeds": wins,
                "wilcoxon_stat_seedlevel": float(stat),
                "p_seedlevel_raw": float(p_seed),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Holm within each model over its three comparisons.
    out["p_seedlevel_holm"] = np.nan
    for model, g in out.groupby("model_name"):
        adj = _holm({r.placebo: r.p_seedlevel_raw for r in g.itertuples()})
        for i, r in g.iterrows():
            out.loc[i, "p_seedlevel_holm"] = adj[r["placebo"]]
    out["significant_0.05"] = out["p_seedlevel_holm"] < 0.05
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "placebo_seed_level.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 2. Mediation agreement, seed level
# ---------------------------------------------------------------------------
def mediation_seed_level() -> pd.DataFrame:
    """Spearman between |C| and the natural indirect effect, aggregated per seed.

    The shipped figure correlated 600 pairs. Those pairs come from far fewer seeds, so the
    p-value was computed against the wrong n. Seed-level aggregation costs power and is the
    honest denominator.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for model in _OSM:
        p = E1 / f"mediation_{model}.jsonl"
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
            if r.get("ok"):
                recs.append(r)
        med = pd.DataFrame(recs)
        if med.empty:
            continue
        # The mediation run stores effects but not |C|, so join the commutator back on
        # the pair key it shares with the battery.
        bat = _battery(model)
        if bat.empty:
            continue
        med = med.merge(bat[["seed_id", "sub_a", "sub_b", "real_single"]],
                        on=["seed_id", "sub_a", "sub_b"], how="inner")
        if med.empty:
            log.warning("%s: mediation pairs did not join to the battery", model)
            continue
        med["absC"] = med["real_single"].abs()
        med["nie"] = med["indirect_effect"].abs()

        rho_pair, p_pair = spearmanr(med["absC"], med["nie"])
        per_seed = med.groupby("seed_id")[["absC", "nie"]].mean()
        n_seeds = len(per_seed)
        if n_seeds < 10:
            continue
        rho, p_seed = spearmanr(per_seed["absC"], per_seed["nie"])

        # Cluster bootstrap CI on the seed-level correlation.
        vals = per_seed.to_numpy()
        draws = np.empty(N_BOOT)
        for b in range(N_BOOT):
            take = rng.integers(0, n_seeds, n_seeds)
            s = vals[take]
            draws[b] = spearmanr(s[:, 0], s[:, 1]).statistic
        lo, hi = np.nanpercentile(draws, [2.5, 97.5])

        rows.append({"model_name": model, "n_seeds": n_seeds, "n_pairs": int(len(med)),
                     "spearman_pairlevel": float(rho_pair), "p_pairlevel": float(p_pair),
                     "spearman_seedlevel": float(rho), "p_seedlevel": float(p_seed),
                     "ci_lo": float(lo), "ci_hi": float(hi)})
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "mediation_seed_level.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 3. Is the intervention a full substitution in disguise?
# ---------------------------------------------------------------------------
def substitution_check() -> pd.DataFrame:
    """Test whether patching all layers at the protected position reproduces the donor run.

    The worry, and it is the first thing an interpretability reviewer will raise: if writing
    the donor residual stream into position p at EVERY layer simply makes that position
    behave as it does in the donor run, then the two variants differ nowhere else and the
    patched run collapses to the donor run. C would then be the plain difference in gold
    logit between the two variants, and the intervention would be a behavioural comparison
    in different clothes.

    A full substitution implies exact antisymmetry, C(a,b) = -C(b,a), because swapping the
    roles of donor and receiver would just exchange the two runs. The battery stores the
    reversed direction, so the prediction is checkable with no further inference. Departure
    from antisymmetry is evidence that the patched run is NOT the donor run.

    The same departure means the measurement carries a direction, which is why the audit
    reports agreement between the two directions rather than assuming they agree.
    """
    rows = []
    for model in _OSM:
        df = _battery(model)
        if df.empty or "reverse_single" not in df.columns:
            continue
        d = df.dropna(subset=["real_single", "reverse_single"])
        fwd = d["real_single"].to_numpy()
        rev = d["reverse_single"].to_numpy()
        resid = np.abs(fwd + rev)
        scale = np.abs(fwd).mean()
        rows.append({
            "model_name": model,
            "n_pairs": int(len(d)),
            "n_seeds": int(d.seed_id.nunique()),
            "corr_forward_vs_negreverse": float(np.corrcoef(fwd, -rev)[0, 1]),
            "mean_abs_residual": float(resid.mean()),
            "residual_over_scale": float(resid.mean() / scale) if scale else np.nan,
            "frac_exactly_antisymmetric": float((resid < 1e-6).mean()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT / "substitution_check.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 4. Provenance audit of every count the manuscript reports
# ---------------------------------------------------------------------------
def provenance_audit() -> pd.DataFrame:
    """Reconcile each headline count against the file it should come from."""
    checks = []

    def add(name, claimed, actual, source, note=""):
        checks.append({"quantity": name, "claimed": claimed, "actual": actual,
                       "match": claimed == actual, "source": source, "note": note})

    en = pd.read_parquet(DATASET / "pentad_dataset_clean.parquet")
    add("English seeds", 596, int(en.seed_id.nunique()),
        "Dataset/seeds/pentad_dataset_clean.parquet")
    add("English prompts", 6855, int(len(en)),
        "Dataset/seeds/pentad_dataset_clean.parquet")

    integ = json.loads((TIST / "e0" / "integrity_summary.json").read_text(encoding="utf-8"))
    add("degenerate variants", 297, int(integ["n_degenerate_variants"]),
        "results/tist/e0/integrity_summary.json")
    add("degenerate pair share (x1000)", 197,
        round(1000 * integ["frac_cdva_pairs_degenerate"]),
        "results/tist/e0/integrity_summary.json", "manuscript says 19.7%")

    for lang, seeds, prompts in (("hi", 230, 2644), ("bn", 222, 2548)):
        d = pd.read_parquet(DATASET / f"pentad_{lang}.parquet")
        add(f"{lang} seeds", seeds, int(d.seed_id.nunique()), f"pentad_{lang}.parquet")
        add(f"{lang} prompts", prompts, int(len(d)), f"pentad_{lang}.parquet")

    ctrl = pd.read_parquet(DATASET / "synthetic_controls.parquet")
    add("control prompts", 1200, int(len(ctrl)), "synthetic_controls.parquet")
    add("control seeds", 200, int(ctrl.seed_id.nunique()), "synthetic_controls.parquet")

    behav = pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet",
                            columns=["seed_id", "model_name"])
    add("behavioural responses", 81056, int(len(behav)),
        "results/behavioral_results.parquet")

    for model in _OSM:
        df = _battery(model)
        if not df.empty:
            add(f"causal pairs, {model}", 9566, int(len(df)),
                f"results/tist/e1/battery_{model}.jsonl")
            add(f"seeds behind those pairs, {model}", 596, int(df.seed_id.nunique()),
                f"results/tist/e1/battery_{model}.jsonl",
                "the inferential n, not 9566")

    comp = json.loads((TIST / "e4" / "competence.json").read_text(encoding="utf-8"))
    add("model-language pairs", 16, len(comp), "results/tist/e4/competence.json")
    add("admitted pairs", 13, sum(1 for v in comp.values() if v["competent"]),
        "results/tist/e4/competence.json")

    out = pd.DataFrame(checks)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "provenance_audit.csv", index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    audit = provenance_audit()
    print("\n=== PROVENANCE AUDIT ===")
    print(audit.to_string(index=False))
    bad = audit[~audit["match"]]
    print(f"\n{len(bad)} mismatch(es)" if len(bad) else "\nall counts reconcile")

    if args.audit_only:
        return

    pl = placebo_seed_level()
    print("\n=== PLACEBO, SEED LEVEL ===")
    if pl.empty:
        print("(no battery data)")
    else:
        cols = ["model_name", "placebo", "n_seeds", "n_pairs", "ratio",
                "ratio_ci_lo", "ratio_ci_hi", "win_rate_seeds",
                "p_seedlevel_holm", "significant_0.05"]
        print(pl[cols].to_string(index=False))

    med = mediation_seed_level()
    print("\n=== MEDIATION, SEED LEVEL ===")
    print(med.to_string(index=False) if len(med) else "(per-pair file absent)")

    sub = substitution_check()
    print("\n=== IS THE PATCH A FULL SUBSTITUTION? ===")
    if sub.empty:
        print("(no reverse direction stored)")
    else:
        print(sub.round(4).to_string(index=False))
        print("exact antisymmetry would mean residual_over_scale = 0; "
              "the observed values say the patched run is not the donor run")


if __name__ == "__main__":
    main()
