"""
File: CPU_Only/predictive_validity.py
Purpose: Predictive validity classifier -- trains on BBQ+CrowS+StereoSet,
         tests on WinoBias. Optionally applied to HolisticBias.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
    -- predictive validity as Killer #3.
  - Zhao et al. (2018). "WinoBias." NAACL 2018.
  - Stanovsky et al. (2019). "Evaluating Gender Bias in Machine Translation."
    ACL 2019.
  - Webster et al. (2020). "Measuring and Reducing Gendered Correlations."
    arXiv:2010.06032
  - Smith et al. (2022). "I Am What I Am: Measuring Biases in LLMs."
    Findings of NAACL 2022. (HolisticBias) -- TODO: verify citation
  - scikit-learn: https://scikit-learn.org

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, ensure_dirs
from CPU_Only.leaderboard import FAILURE_MODES

logger = logging.getLogger(__name__)

_TRAIN_SOURCES = {"bbq", "crows_pairs", "stereoset"}
_TEST_SOURCE = "winobias"  # held-out test set


def _build_feature_matrix(behavioral_df: pd.DataFrame, cdva_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-(seed_id, model_name) feature vector.
    Features: slot pass rates, CDVA score, CoT robustness flag.
    """
    rows: list[dict] = []
    seed_model_pairs = (
        behavioral_df[["seed_id", "model_name", "seed_source", "seed_category"]]
        .drop_duplicates(subset=["seed_id", "model_name"])
    )

    for _, pair in seed_model_pairs.iterrows():
        seed_id = pair["seed_id"]
        model_name = pair["model_name"]
        b_rows = behavioral_df[
            (behavioral_df["seed_id"] == seed_id)
            & (behavioral_df["model_name"] == model_name)
            & (behavioral_df["sample_index"] == 0)
        ]

        def _rate(slot: str) -> float:
            s = b_rows[b_rows["slot"] == slot]
            if s.empty:
                return 0.0
            return float((s["success_flag"] & (s["parsed_answer"] != "")).mean())  # type: ignore

        a_rate = _rate("a")
        b_rate = _rate("b")
        c_rate = _rate("c")
        d_rate = _rate("d")
        e_rate = _rate("e")

        # CoT robustness: 1 if majority vote constant
        e_rows = b_rows[b_rows["slot"] == "e"]
        if len(e_rows) >= 2:
            vc = e_rows["parsed_answer"].value_counts()
            cot_robust = 1.0 if (len(vc) > 0 and vc.iloc[0] > len(e_rows) / 2) else 0.0
        else:
            cot_robust = 0.0

        # CDVA score
        cdva_seed = cdva_df[
            (cdva_df["seed_id"] == seed_id)
            & (cdva_df["model_name"] == model_name)
            & (cdva_df["success_flag"] == True)  # noqa: E712
        ] if len(cdva_df) > 0 else pd.DataFrame()
        cdva_score = float(cdva_seed["cdva_pair_score"].mean()) if len(cdva_seed) > 0 else 0.0

        rows.append(
            {
                "seed_id": seed_id,
                "model_name": model_name,
                "seed_source": pair["seed_source"],
                "seed_category": pair["seed_category"],
                "feat_a_rate": a_rate,
                "feat_b_rate": b_rate,
                "feat_c_rate": c_rate,
                "feat_d_rate": d_rate,
                "feat_e_rate": e_rate,
                "feat_cot_robust": cot_robust,
                "feat_cdva_score": cdva_score,
            }
        )

    return pd.DataFrame(rows)


def _compute_fm_labels(
    behavioral_df: pd.DataFrame,
    cdva_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-(seed_id, model_name) binary failure-mode labels."""
    from CPU_Only.leaderboard import (
        _fm1_proxy_substitution,
        _fm2_arch_indistinguishable,
        _fm3_context_blindness,
        _fm4_criterion_leakage,
        _fm5_approx_ceiling,
    )

    rows: list[dict] = []
    for seed_id in behavioral_df["seed_id"].unique():
        seed_rows = behavioral_df[behavioral_df["seed_id"] == seed_id]
        seed_cdva = cdva_df[cdva_df["seed_id"] == seed_id] if len(cdva_df) > 0 else pd.DataFrame()
        for model_name in seed_rows["model_name"].unique():
            m_rows = seed_rows[seed_rows["model_name"] == model_name]
            rows.append(
                {
                    "seed_id": seed_id,
                    "model_name": model_name,
                    "FM1": int(_fm1_proxy_substitution(m_rows) > 0.5),
                    "FM2": int(_fm2_arch_indistinguishable(m_rows, seed_cdva) > 0.5),
                    "FM3": int(_fm3_context_blindness(m_rows) > 0.5),
                    "FM4": int(_fm4_criterion_leakage(m_rows) > 0.5),
                    "FM5": int(_fm5_approx_ceiling(m_rows) > 0.5),
                }
            )
    return pd.DataFrame(rows)


def run_predictive_validity(
    behavioral_df: pd.DataFrame,
    cdva_df: pd.DataFrame,
) -> dict:
    """
    Train on BBQ+CrowS+StereoSet, evaluate on WinoBias.

    Returns
    -------
    dict
        Per-FM metrics: accuracy, F1, ROC-AUC.
    """
    ensure_dirs()
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore

    features_df = _build_feature_matrix(behavioral_df, cdva_df)
    labels_df = _compute_fm_labels(behavioral_df, cdva_df)
    merged = features_df.merge(labels_df, on=["seed_id", "model_name"])

    feat_cols = [c for c in merged.columns if c.startswith("feat_")]

    train = merged[merged["seed_source"].isin(_TRAIN_SOURCES)]
    test = merged[merged["seed_source"] == _TEST_SOURCE]

    if len(test) == 0:
        logger.warning("No WinoBias test rows found. Predictive validity skipped.")
        return {}

    X_train = train[feat_cols].fillna(0).values
    X_test = test[feat_cols].fillna(0).values

    results: dict = {}
    for fm in FAILURE_MODES:
        y_train = train[fm].values
        y_test = test[fm].values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning("FM %s has only one class in train or test; skipping.", fm)
            continue

        clf = LogisticRegression(max_iter=500, random_state=20260101)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except Exception:
            auc = float("nan")

        results[fm] = {"accuracy": acc, "f1": f1, "roc_auc": auc}
        logger.info(
            "Predictive validity | %s | acc=%.3f f1=%.3f auc=%.3f",
            fm, acc, f1, auc,
        )

    return results
