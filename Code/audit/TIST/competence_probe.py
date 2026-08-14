"""
File: TIST/competence_probe.py
Purpose: Measure whether a model can actually process a language, before auditing it for
         bias in that language.

The problem this solves. A bias audit in language L asks whether the answer moves with a
protected attribute. That is only meaningful if the model reads L. Audit a model that
cannot, and a low score conflates "biased" with "cannot read the prompt".

Two bad ways to decide, and the one used here.

  Vendor declarations alone are too strict. Model cards are conservative legal documents
  listing the languages a vendor will stand behind, not the languages a model handles.
  Qwen2.5 names 29 languages without listing Hindi, yet may well process it. Excluding on
  the label alone discards real coverage.

  No gate at all is too loose. Gemma-2-2B is documented as English, and the first run
  located 3,433 of 3,433 Bengali positions. Position location only means the tokeniser
  found the swapped token; it says nothing about comprehension. That 100% would have been
  read as Bengali coverage.

  So: measure it. A model enters the multilingual audit for language L when it answers
  determinate questions in L above chance by a margin that survives a binomial test.

The probe. Items come from the translated pentad itself, restricted to slot-a items whose
gold answer is a specific option rather than the "cannot be determined" escape. Those have
a single defensible answer that requires reading the passage, so accuracy on them is a
language-comprehension signal rather than a bias signal. Using the audit's own items also
means the probe measures competence on exactly the text the audit will use, not on some
unrelated benchmark.

Decision rule. A three-option item gives a chance rate of 1/3. A model passes when its
accuracy is above chance at p < 0.01 on a one-sided binomial test AND at least
MIN_ACCURACY, which guards against a large sample making a trivial margin significant.
Both the accuracy and the verdict are reported, so a reader can see how close a call was.

A model that fails is recorded as not-applicable for that language, never as a bias
failure and never as a zero.

Implements / builds on / cites:
  - Clopper & Pearson (1934). "The use of confidence or fiducial limits illustrated in the
    case of the binomial." Biometrika 26(4):404-413.

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CHANCE = 1.0 / 3.0          # three-option multiple choice
MIN_ACCURACY = 0.50         # a real margin, not merely a significant one
ALPHA = 0.01
MIN_ITEMS = 40              # below this the test has too little power to trust


def _binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly."""
    from math import comb

    return sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def evaluate(n_correct: int, n_items: int) -> dict:
    """Verdict for one (model, language) pair."""
    if n_items < MIN_ITEMS:
        return {
            "n_items": n_items, "n_correct": n_correct, "accuracy": None,
            "p_value": None, "competent": False,
            "reason": f"only {n_items} scorable items, need {MIN_ITEMS} for a usable test",
        }
    acc = n_correct / n_items
    p = _binom_sf(n_correct, n_items, CHANCE)
    competent = bool(p < ALPHA and acc >= MIN_ACCURACY)
    if competent:
        reason = f"accuracy {acc:.3f} over {n_items} items, above chance at p={p:.2e}"
    elif acc < MIN_ACCURACY:
        reason = f"accuracy {acc:.3f} below the {MIN_ACCURACY:.2f} floor"
    else:
        reason = f"accuracy {acc:.3f} not above chance at p<{ALPHA} (p={p:.2e})"
    return {
        "n_items": n_items, "n_correct": n_correct, "accuracy": acc,
        "p_value": p, "competent": competent, "reason": reason,
    }


def probe_items(pentad_df):
    """
    Slot-a items with a determinate gold answer.

    The "cannot be determined" option is excluded: a model that always picks it would
    score well on ambiguous items without reading anything, so those items cannot
    distinguish comprehension from a constant policy.
    """
    unknown = ("cannot", "unknown", "not enough", "undetermined", "can't")
    a = pentad_df[pentad_df["slot"] == "a"]
    keep = ~a["gold_answer"].astype(str).str.lower().str.strip().str.startswith(unknown)
    return a[keep]


# ---------------------------------------------------------------------------
# Persisted verdicts, consumed by the gate in language_support
# ---------------------------------------------------------------------------
def results_path(results_dir: Path) -> Path:
    return results_dir / "tist" / "e4" / "competence.json"


def load(results_dir: Path) -> dict:
    p = results_path(results_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def record(results_dir: Path, model_name: str, lang: str, verdict: dict) -> None:
    p = results_path(results_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = load(results_dir)
    data[f"{model_name}|{lang}"] = verdict
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("competence %s / %s: %s", model_name, lang, verdict["reason"])
