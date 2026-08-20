# E0 — item integrity: degenerate slot-c counterfactuals

Found while building the multilingual pentad. It bears directly on whether the commutator
measures a causal property, so it is reported here as its own experiment.
Produced by `Code/audit/TIST/e0_item_integrity.py`. No model was run. Artifacts:
`integrity_summary.json`, `degenerate_variants.csv`, `cdva_degeneracy_mask.parquet`,
`exclusion_log.csv`, and the cleaned pentad at
`Dataset/seeds/pentad_dataset_clean.parquet`.

---

## 1. The defect

Slot c substitutes an admissible value of the protected attribute into the seed. When the
substituted term is already the text of another answer option, the item offers the same
option twice:

```
(A) The chemist   (B) Cannot answer   (C) The chemist
```

A model choosing between two identical options is not displaying demographic sensitivity.
Any logit movement such an item produces is item construction, not bias.

| Quantity | Value |
|---|---|
| Degenerate slot-c variants | 297 of 2,980 (9.97%) |
| Seeds touched | 286 of 596 (48.0%) |
| CDVA pairs involving a degenerate variant | 4,708 of 23,840 (19.75%) |
| By source | CrowS-Pairs 169, BBQ 126, StereoSet 2 |

## 2. It inflates the measured causal signal on every model

Mean |C| on pairs that touch a degenerate variant against pairs that do not:

| Model | Clean pairs | Degenerate pairs | Reported (all pairs) | Inflation |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 0.925 | 1.332 | 1.006 | +0.080 |
| Llama-3.1-8B-Instruct | 0.468 | 0.688 | 0.511 | +0.043 |
| Phi-4-mini-Instruct | 0.550 | 0.650 | 0.569 | +0.020 |
| Gemma-2-2B-it | 0.502 | 0.523 | 0.506 | +0.004 |

The direction is the same on all four models and the ordering of the inflation follows the
severity ordering. Roughly a fifth of the pairs behind the published severity numbers came
from malformed items, and those pairs carry 44% higher |C| on Qwen and 47% higher on Llama.

## 3. Repair

Exclusion, not substitution. An earlier attempt redrew the colliding term from
`Dataset/equivalence_sets.yaml` and failed: the colliding strings are BBQ options naming
people by role ("The chemist", "The grandson", "The one who is pregnant"), so no
equivalence set contains them, every redraw hit the union fallback, and "The chemist"
became "Sikh". Manufacturing a counterfactual the item was never built around is worse
than dropping it.

Dropping the 297 variants costs less than feared. All 596 seeds keep at least three slot-c
variants, which is the minimum `CPU_Only/scoring.compute_mirage_b` needs for the stability
check, so **no seed is lost**. The cleaned pentad holds 6,855 prompts against 7,152.

Exclusion also needs no new inference: the affected CDVA pairs are masked out of the
existing results rather than recomputed.

## 4. What this means for the TIST manuscript

This is the strongest single item of evidence on item integrity, and it should be framed
that way. The open question is whether the commutator measures a causal property or an
artefact of patching. Here is a case where part of it demonstrably was an artefact, found
by the instrument's own bookkeeping, quantified, removed, and the audit rerun. An
auditing system that can locate a construction fault in its own item pool is a stronger
claim than one that reports only clean numbers.

Required changes:

1. Every severity, commutativity and validity-gap number recomputes on clean pairs.
2. The paper reports both figures and the difference, in a table.
3. The multilingual sets are built from the cleaned pentad, so the defect is not exported
   to Hindi and Bengali.
4. The system architecture section gains an item-integrity stage, since this check should
   run before any audit, not after.

## 5. Related

Two other integrity items came out of Phase 1. See `../e2/FINDINGS.md` for the threshold
unit mismatch, which is independent of this one and also moves the headline numbers.
