"""
File: TIST/e0_item_integrity.py
Purpose: E0 -- detect and repair degenerate slot-c items in the pentad, and quantify
         how much of the measured causal signal they account for.

Why this exists. The audit's headline claim rests on the causal reading of
CDVA was not established. One concrete way that reading can fail is if part of the
commutator comes from item construction rather than from the protected attribute. This
module found exactly that case.

The defect. Slot c substitutes an admissible value of the protected attribute into the
seed. When the substituted term is already the text of another answer option, the item
ends up offering the same option twice:

    (A) The chemist   (B) Cannot answer   (C) The chemist

297 of 2,980 slot-c variants (9.97%) are degenerate in this way, touching 286 of 596
seeds. They participate in 4,708 of 23,840 CDVA pairs (19.75%), and those pairs carry
systematically larger |C| on every model. On Qwen2.5-7B the mean |C| is 1.332 on
degenerate pairs against 0.925 on clean ones. A model choosing between two identical
options is not exhibiting demographic sensitivity, so that excess is measurement
artefact.

What this module does.

  1. DETECT. Flag every slot-c variant whose option list contains a repeat, and emit the
     clean-pair mask that downstream analysis uses.
  2. QUANTIFY. Report |C| and the audit aggregates with and without the degenerate pairs,
     so the paper can state the size of the artefact instead of asserting it is small.
  3. REPAIR BY EXCLUSION. Drop the degenerate variant from the seed's slot-c set and emit
     a clean pentad, then report how many seeds retain enough slot-c variants for the
     stability check to remain well defined.

Why exclusion rather than substitution. An earlier version redrew the colliding term from
the equivalence sets in Dataset/equivalence_sets.yaml. That failed: the colliding strings
are BBQ answer options naming people by role ("The chemist", "The grandson"), not
protected-attribute values, so no equivalence set contains them and every redraw fell back
to the union of all sets, rewriting "The chemist" as "Sikh". Substituting a term the item
was never built around manufactures a counterfactual rather than repairing one. Dropping
the malformed variant keeps the remaining counterfactuals exactly as generated and states
the loss openly, which is the defensible move for a paper about measurement validity.

An added benefit is that exclusion needs no new inference. The affected CDVA pairs are
masked out of the existing results rather than recomputed.

Implements / builds on / cites:
  - Parrish et al. (2022). "BBQ." Findings of ACL 2022 -- source of the option structure.
  - Blodgett et al. (2021). "Stereotyping Norwegian Salmon: An Inventory of Pitfalls in
    Fairness Benchmark Datasets." ACL 2021 -- the class of construction fault this is.

Usage:
  python TIST/e0_item_integrity.py            # detect + quantify + write repaired pentad
  python TIST/e0_item_integrity.py --no-repair

Part of the MIRAGE audit codebase.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, SEEDS_DIR  # noqa: E402

log = logging.getLogger("e0")

OUT = RESULTS_DIR / "tist" / "e0"
_OSM = [m["name"] for m in OSM_MODELS]
_EQ_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "equivalence_sets.yaml"

_OPT_RE = re.compile(r"\(([ABC])\)\s*(.+)")


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def options_of(prompt_text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2).strip()) for m in _OPT_RE.finditer(prompt_text or "")]


def is_degenerate(prompt_text: str) -> bool:
    opts = [t.lower() for _, t in options_of(prompt_text)]
    return bool(opts) and len(opts) != len(set(opts))


def flag_pentad(pentad: pd.DataFrame) -> pd.DataFrame:
    df = pentad.copy()
    df["degenerate"] = False
    mask = df["slot"] == "c"
    df.loc[mask, "degenerate"] = df.loc[mask, "prompt_text"].map(is_degenerate)
    return df


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------
def quantify(flagged: pd.DataFrame, cdva: pd.DataFrame) -> dict:
    deg = set(
        zip(
            flagged.loc[flagged["degenerate"], "seed_id"],
            flagged.loc[flagged["degenerate"], "subvariant"],
        )
    )
    cd = cdva.copy()
    cd["touches_degenerate"] = [
        (s, a) in deg or (s, b) in deg
        for s, a, b in zip(cd["seed_id"], cd["pair_A_subvariant"], cd["pair_B_subvariant"])
    ]
    cd["absC"] = cd["delta_logit"].abs()

    per_model = {}
    for m, g in cd.groupby("model_name"):
        clean, dirty = g[~g["touches_degenerate"]], g[g["touches_degenerate"]]
        per_model[m] = {
            "n_pairs_total": int(len(g)),
            "n_pairs_degenerate": int(len(dirty)),
            "mean_absC_all": float(g["absC"].mean()),
            "mean_absC_clean": float(clean["absC"].mean()),
            "mean_absC_degenerate": float(dirty["absC"].mean()) if len(dirty) else None,
            "severity_inflation": (
                float(g["absC"].mean() - clean["absC"].mean()) if len(clean) else None
            ),
        }

    slot_c = flagged[flagged["slot"] == "c"]
    summary = {
        "n_slot_c_variants": int(len(slot_c)),
        "n_degenerate_variants": int(slot_c["degenerate"].sum()),
        "frac_degenerate_variants": float(slot_c["degenerate"].mean()),
        "n_seeds_total": int(flagged["seed_id"].nunique()),
        "n_seeds_touched": int(slot_c.loc[slot_c["degenerate"], "seed_id"].nunique()),
        "degenerate_by_source": slot_c[slot_c["degenerate"]]["seed_source"]
        .value_counts()
        .to_dict(),
        "n_cdva_pairs_total": int(len(cd)),
        "n_cdva_pairs_degenerate": int(cd["touches_degenerate"].sum()),
        "frac_cdva_pairs_degenerate": float(cd["touches_degenerate"].mean()),
        "per_model": per_model,
    }
    return summary, cd


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------
_MIN_C_VARIANTS = 3  # CPU_Only/scoring.compute_mirage_b requires at least this many


def repair(flagged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop every degenerate slot-c variant and report the cost to each seed.

    Nothing else is altered. Slots a, b, d and e pass through untouched, and the
    surviving slot-c variants keep the text the generator produced.
    """
    clean = flagged[~flagged["degenerate"]].copy()

    before = (
        flagged[flagged["slot"] == "c"].groupby("seed_id").size().rename("n_c_before")
    )
    after = clean[clean["slot"] == "c"].groupby("seed_id").size().rename("n_c_after")
    counts = pd.concat([before, after], axis=1).fillna(0).astype(int)
    counts["n_dropped"] = counts["n_c_before"] - counts["n_c_after"]
    counts["below_min"] = counts["n_c_after"] < _MIN_C_VARIANTS

    log_rows = counts.reset_index().to_dict("records")
    log.info(
        "dropped %d degenerate slot-c variants; %d seeds lost at least one; "
        "%d seeds fall below the %d-variant minimum for the stability check",
        int(counts["n_dropped"].sum()),
        int((counts["n_dropped"] > 0).sum()),
        int(counts["below_min"].sum()),
        _MIN_C_VARIANTS,
    )
    assert not clean.loc[clean["slot"] == "c", "prompt_text"].map(is_degenerate).any()
    return clean, pd.DataFrame(log_rows)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    pentad = pd.read_parquet(SEEDS_DIR / "pentad_dataset.parquet")
    cdva = pd.read_parquet(RESULTS_DIR / "cdva_results.parquet")
    cdva = cdva[cdva["success_flag"] == True]  # noqa: E712

    flagged = flag_pentad(pentad)
    summary, cd = quantify(flagged, cdva)

    (OUT / "integrity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cd[["seed_id", "model_name", "pair_A_subvariant", "pair_B_subvariant", "absC", "touches_degenerate"]].to_parquet(
        OUT / "cdva_degeneracy_mask.parquet", index=False
    )
    flagged[flagged["degenerate"]][
        ["seed_id", "seed_source", "seed_category", "subvariant", "swap_token", "prompt_text"]
    ].to_csv(OUT / "degenerate_variants.csv", index=False, encoding="utf-8")

    log.info(
        "degenerate slot-c variants: %d/%d (%.2f%%), seeds touched %d/%d, cdva pairs %d/%d (%.2f%%)",
        summary["n_degenerate_variants"],
        summary["n_slot_c_variants"],
        100 * summary["frac_degenerate_variants"],
        summary["n_seeds_touched"],
        summary["n_seeds_total"],
        summary["n_cdva_pairs_degenerate"],
        summary["n_cdva_pairs_total"],
        100 * summary["frac_cdva_pairs_degenerate"],
    )

    if not args.no_repair:
        clean, rlog = repair(flagged)
        clean.to_parquet(SEEDS_DIR / "pentad_dataset_clean.parquet", index=False)
        rlog.to_csv(OUT / "exclusion_log.csv", index=False, encoding="utf-8")
        summary["exclusion"] = {
            "n_variants_dropped": int(rlog["n_dropped"].sum()),
            "n_seeds_affected": int((rlog["n_dropped"] > 0).sum()),
            "n_seeds_below_min_c_variants": int(rlog["below_min"].sum()),
            "min_c_variants_required": _MIN_C_VARIANTS,
            "n_prompts_after": int(len(clean)),
        }
        (OUT / "integrity_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
