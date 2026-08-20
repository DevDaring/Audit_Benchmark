# E3 — inter-annotator agreement for pentad slots (d) and (e)

Human validation of the generated slots. Produced by `Code/audit/TIST/e3_annotation.py` over a
200-item stratified sample of generator-drafted items (87 slot (d), 113 slot (e), 61
strata by benchmark x bias axis x subvariant). Guidelines: `annotation_guidelines.md`.
Artifacts: `iaa_sample_manifest.csv`, `iaa_sheet_*.csv`, `llm_annotations.jsonl`,
`iaa_report.json`.

---

## 1. Human annotation: outstanding

`iaa_sheet_annotator_a.csv` and `iaa_sheet_annotator_b.csv` are prepared and unfilled. Cohen's
kappa between the two human annotators is the headline number the paper needs and it
cannot be computed until both are returned. The scoring command reports the shortfall
rather than substituting the LLM panel.

## 2. LLM panel: three annotators, three vendors

DeepSeek-chat, gpt-4o-mini via OpenRouter, and mistral-small, all at temperature 0, on
the same 200 items under the same guidelines.

The panel originally included Gemini-2.5-Flash. All four keys returned HTTP 429 for the
entire pass and it contributed no labels, so it was replaced by gpt-4o-mini. That
replacement is also the better choice on principle: Gemini-2.5-Flash is one of the eight
models under audit, and an evaluated system should not grade the items it is later scored
on.

**Fleiss' kappa on the overall pass/fail judgement: 0.219** over 200 items, which is
"fair" on the Landis and Koch scale. Pass rates diverge sharply: DeepSeek 0.835,
mistral-small 0.710, gpt-4o-mini 0.395.

## 3. The disagreement is entirely in the semantic criteria

| Criterion | Fleiss' kappa | Unanimous agreement | Pass-rate spread |
|---|---|---|---|
| C1 gold-answer invariance | +0.365 | 0.705 | 0.155 |
| C2 no new information | +0.064 | 0.455 | 0.515 |
| C3 grammaticality | -0.003 | 0.990 | 0.010 |
| C4 structural integrity | -0.005 | 0.985 | 0.015 |

Two readings matter, and the second corrects a trap.

**The contested criterion is C2.** The three annotators agree unanimously on only 45.5%
of items, and gpt-4o-mini passes 41% where DeepSeek passes 92%. Judging whether an added
sentence smuggles in a new fact is the judgement the panel cannot make consistently. C1
is intermediate at 70.5% unanimous.

**C3 and C4 are not disagreements.** Their kappa values sit at roughly zero, but the
annotators agree on 99.0% and 98.5% of items respectively. When a criterion is
near-unanimous there is almost no variance in the labels, chance agreement approaches
observed agreement, and kappa collapses toward zero regardless of how well the raters
actually agree. This is the kappa paradox (Feinstein and Cicchetti, 1990). Reading these
rows as disagreement would invert their meaning, so the report selects the contested
criterion by observed disagreement and flags the paradox cases explicitly.

## 4. What this means for the TIST manuscript

The result is more useful than a single agreement number would have been.

1. **Mechanical criteria automate; semantic ones do not.** Grammaticality and structural
   integrity reach 98 to 99% unanimous agreement across three vendors, so the system can
   enforce them automatically as a pipeline stage. Answer preservation and the no-new-
   information constraint cannot be delegated: agreement on C2 is barely above chance.
2. **This is the argument for human annotation**, made with evidence rather than
   assertion. On whether a formal protocol exists, the answer is that a
   protocol exists, that its mechanical half is automated in the released toolkit, and
   that its semantic half needs human judgement because a three-vendor LLM panel splits
   on it.
3. **Report kappa with raw agreement throughout.** The C3 and C4 rows would otherwise be
   misread by a reader skimming the table.

Placement: Methods, in the pentad-construction subsection, with the full table in the
appendix. The automatable checks belong in the system architecture section as an
item-integrity stage, alongside the degenerate-option check from E0.

## 5. Blocked

Cohen's kappa between the two human annotators, the agreement of the LLM majority with
the human consensus, and the rescoring of items that fail the human majority. All three
resume with `python TIST/e3_annotation.py score` once the sheets are returned.

## Reference

Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The
problems of two paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543-549.
