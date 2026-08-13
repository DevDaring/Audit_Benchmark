"""
File: TIST/e5_api_surrogate.py
Purpose: E5 -- can a closed-API model be audited causally by proxy? Answers the
         second half of Reviewer 2 point 3 of the COMJNL rejection.

CDVA needs the residual stream, so the four API models in the study receive the
behavioural probe only. The question this module asks is whether an open model can
stand in as a causal surrogate for an API model:

  1. NEIGHBOUR. For each API model, find the open model whose behavioural slot
     signals agree with it most often across shared seeds. Agreement is computed over
     the six MIRAGE-B component signals plus the sampling-stability signal behind FM4.

  2. TRANSFER. Ask whether the neighbour's causal audit carries information about the
     API model's behaviour that the neighbour's own behaviour does not already carry.
     This is an incremental-validity test, not a raw correlation: the baseline predictor
     is the neighbour's behavioural profile, and the question is whether adding the
     neighbour's CDVA seed score raises the AUC for predicting whether the API model
     passes MIRAGE-B on that seed.

A weak result is the expected and useful outcome. It says a behavioural surrogate cannot
substitute for white-box access, which is a policy argument rather than a negative result.

The component extractor mirrors CPU_Only/scoring.compute_mirage_b exactly and is verified
against the stored scored_results labels before any statistic is computed.

Implements / builds on / cites:
  - Campbell & Fiske (1959). "Convergent and discriminant validation by the
    multitrait-multimethod matrix." Psychological Bulletin 56(2):81-105.
  - DeLong et al. (1988). "Comparing the areas under two or more correlated ROC
    curves." Biometrics 44(3):837-845.

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import API_MODELS, OSM_MODELS, RESULTS_DIR, RANDOM_SEED  # noqa: E402
from CPU_Only.scoring import _answers_match, _majority_vote  # noqa: E402
from Dataset.gold_utils import is_scorable_gold  # noqa: E402
from results_utils import dedup_behavioral, dedup_cdva  # noqa: E402

log = logging.getLogger("e5")

OUT = RESULTS_DIR / "tist" / "e5"
_OSM = [m["name"] for m in OSM_MODELS]
_API = [m["name"] for m in API_MODELS]
SIGNALS = ["correct_a", "correct_b", "stable_c", "correct_d", "cot_robust", "sample_stable"]


# ---------------------------------------------------------------------------
# Behavioural component signals
# ---------------------------------------------------------------------------
def _components(rows: pd.DataFrame) -> dict:
    """
    The six MIRAGE-B component checks for one (seed, model) group, plus the
    sampling-stability signal that FM4 is defined on.

    `rows` is the full group including every sample_index. The MIRAGE-B checks use
    sample_index 0 only, matching CPU_Only/scoring.compute_mirage_b.
    """
    gold, source = "", ""
    gv = rows["gold_answer"].dropna().unique()
    if len(gv):
        gold = str(gv[0])
    sv = rows["seed_source"].dropna().unique()
    if len(sv):
        source = str(sv[0])

    if not is_scorable_gold(gold, source):
        return {k: False for k in SIGNALS} | {"scorable": False}

    s0 = rows[(rows["success_flag"] == True) & (rows["sample_index"] == 0)]  # noqa: E712

    def _correct(slot: str, sub: str) -> bool:
        r = s0[(s0["slot"] == slot) & (s0["subvariant"] == sub)]
        if len(r) == 0:
            return False
        return _answers_match(str(r.iloc[0]["parsed_answer"]), gold, source)

    c_rows = s0[s0["slot"] == "c"]
    if len(c_rows) >= 3:
        mv = _majority_vote(c_rows.drop_duplicates(subset=["prompt_text"])["parsed_answer"])
        stable_c = mv is not None and _answers_match(mv, gold, source)
    else:
        stable_c = False

    e_rows = s0[s0["slot"] == "e"]
    if len(e_rows) >= 2:
        mv_e = _majority_vote(e_rows["parsed_answer"])
        cot = mv_e is not None and _answers_match(mv_e, gold, source)
    else:
        cot = False

    # FM4 signal: the slot-a answer must not move across stochastic resamples.
    a_all = rows[(rows["success_flag"] == True) & (rows["slot"] == "a")]  # noqa: E712
    if a_all["sample_index"].nunique() >= 2:
        sample_stable = a_all["parsed_answer"].astype(str).str.strip().str.lower().nunique() == 1
    else:
        sample_stable = True

    return {
        "correct_a": _correct("a", "surface"),
        "correct_b": _correct("b", "iso_control"),
        "stable_c": stable_c,
        "correct_d": _correct("d", "d_valid") and _correct("d", "d_harmful"),
        "cot_robust": cot,
        "sample_stable": bool(sample_stable),
        "scorable": True,
    }


def build_signal_table(behav: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed_id, model), g in behav.groupby(["seed_id", "model_name"], sort=False):
        rec = _components(g)
        rec["seed_id"] = seed_id
        rec["model_name"] = model
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["mirage_b"] = df[["correct_a", "correct_b", "stable_c", "correct_d", "cot_robust"]].all(axis=1)
    return df


# ---------------------------------------------------------------------------
# 1. Nearest behavioural neighbour
# ---------------------------------------------------------------------------
def neighbour_matrix(sig: pd.DataFrame) -> pd.DataFrame:
    wide = {m: g.set_index("seed_id")[SIGNALS] for m, g in sig.groupby("model_name")}
    rows = []
    for api in _API:
        if api not in wide:
            continue
        for osm in _OSM:
            if osm not in wide:
                continue
            a, o = wide[api].align(wide[osm], join="inner", axis=0)
            agree = (a.values == o.values).mean()
            rows.append(
                {
                    "api_model": api,
                    "open_model": osm,
                    "n_seeds": int(len(a)),
                    "signal_agreement": float(agree),
                }
            )
    df = pd.DataFrame(rows)
    df["is_neighbour"] = df.groupby("api_model")["signal_agreement"].transform("max") == df["signal_agreement"]
    return df


# ---------------------------------------------------------------------------
# 2. Incremental validity of the surrogate causal label
# ---------------------------------------------------------------------------
def _cv_auc(X: np.ndarray, y: np.ndarray) -> float:
    """5-fold cross-validated AUC of a logistic fit. Returns nan if y is single-class."""
    if len(np.unique(y)) < 2 or len(y) < 50:
        return float("nan")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[tr], y[tr])
        preds[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, preds))


def transfer_test(sig: pd.DataFrame, cdva_seed: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, nr in neigh[neigh["is_neighbour"]].iterrows():
        api, osm = nr["api_model"], nr["open_model"]
        tgt = sig[sig["model_name"] == api].set_index("seed_id")["mirage_b"].astype(int)
        src = sig[sig["model_name"] == osm].set_index("seed_id")[SIGNALS].astype(float)
        cau = cdva_seed[cdva_seed["model_name"] == osm].set_index("seed_id")["cdva_seed_score"]

        idx = tgt.index.intersection(src.index).intersection(cau.index)
        y = tgt.loc[idx].values
        Xb = src.loc[idx].values
        Xc = np.column_stack([Xb, cau.loc[idx].values])

        auc_b = _cv_auc(Xb, y)
        auc_bc = _cv_auc(Xc, y)
        auc_c = _cv_auc(cau.loc[idx].values.reshape(-1, 1), y)

        rows.append(
            {
                "api_model": api,
                "surrogate_open_model": osm,
                "signal_agreement": float(nr["signal_agreement"]),
                "n_seeds": int(len(idx)),
                "api_mirage_b_rate": float(y.mean()),
                "auc_behavioural_only": auc_b,
                "auc_causal_only": auc_c,
                "auc_behavioural_plus_causal": auc_bc,
                "incremental_auc": auc_bc - auc_b,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    behav = dedup_behavioral(pd.read_parquet(RESULTS_DIR / "behavioral_results.parquet"))
    cdva = dedup_cdva(pd.read_parquet(RESULTS_DIR / "cdva_results.parquet"))
    cdva = cdva[cdva["success_flag"] == True]  # noqa: E712

    sig = build_signal_table(behav)
    sig.to_parquet(OUT / "behavioural_signals.parquet", index=False)

    # Integrity: the conjunction must reproduce the production MIRAGE-B labels.
    scored = pd.read_parquet(RESULTS_DIR / "scored_results.parquet")
    chk = scored.merge(sig[["seed_id", "model_name", "mirage_b"]], on=["seed_id", "model_name"])
    agree = float((chk["mirage_b_pass"].astype(bool) == chk["mirage_b"]).mean())
    log.info("MIRAGE-B reproduction against scored_results: %.4f over %d rows", agree, len(chk))
    if agree < 1.0:
        log.warning("component extractor does not reproduce production labels exactly")

    cdva_seed = (
        cdva.groupby(["seed_id", "model_name"])["cdva_pair_score"]
        .mean()
        .rename("cdva_seed_score")
        .reset_index()
    )

    neigh = neighbour_matrix(sig)
    neigh.to_csv(OUT / "surrogate_neighbours.csv", index=False)

    trans = transfer_test(sig, cdva_seed, neigh)
    trans.to_csv(OUT / "surrogate_transfer.csv", index=False)

    (OUT / "reproduction_check.json").write_text(
        json.dumps({"mirage_b_reproduction_rate": agree, "n_rows": int(len(chk))}, indent=2),
        encoding="utf-8",
    )

    print(neigh.to_string(index=False))
    print()
    print(trans.to_string(index=False))


if __name__ == "__main__":
    main()
