# E5 — closed-API pathway: behavioural surrogate feasibility

Open-to-closed surrogate transfer. Produced by
`Code/audit/TIST/e5_api_surrogate.py` from `results/behavioral_results.parquet` and
`results/cdva_results.parquet`. No model was run. Artifacts:
`surrogate_neighbours.csv`, `surrogate_transfer.csv`, `behavioural_signals.parquet`.

Integrity check: the component extractor reproduces the production MIRAGE-B labels in
`scored_results.parquet` on 100% of 4,768 seed x model rows before any statistic is
computed (`reproduction_check.json`).

---

## 1. Nearest behavioural neighbour

Agreement is the mean match rate over six binary signals per seed, across all 596 shared
seeds: the five MIRAGE-B component checks (slot-a correctness, slot-b correctness,
slot-c stability, slot-d correctness, slot-e chain-of-thought robustness) plus the
sampling-stability signal that FM4 is defined on.

| API model | Nearest open model | Signal agreement |
|---|---|---|
| Qwen3-Next-80B-A3B | Qwen2.5-7B-Instruct | 0.704 |
| Amazon Nova-2-Lite | Llama-3.1-8B-Instruct | 0.710 |
| Gemini-2.5-Flash | Qwen2.5-7B-Instruct | 0.733 |
| Mistral-Medium | Llama-3.1-8B-Instruct | 0.701 |

The matching is not arbitrary. Qwen3-Next-80B pairs with the other Qwen model, which is
the one case where the surrogate and the target share a training lineage.

## 2. The causal label does not transfer

Predicting whether the API model passes MIRAGE-B on a seed, 5-fold cross-validated AUC,
logistic fit, 596 seeds:

| API model | Surrogate | Behavioural only | Causal only | Behavioural + causal | Incremental |
|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | Qwen2.5-7B | 0.806 | 0.537 | 0.809 | +0.004 |
| Amazon Nova-2-Lite | Llama-3.1-8B | 0.794 | 0.634 | 0.796 | +0.002 |
| Gemini-2.5-Flash | Qwen2.5-7B | 0.777 | 0.550 | 0.780 | +0.004 |
| Mistral-Medium | Llama-3.1-8B | 0.725 | 0.538 | 0.725 | -0.000 |

Three readings, in order of importance.

1. **Behaviour transfers.** An open model predicts an API model's behavioural outcome at
   AUC 0.72 to 0.81. A behavioural surrogate is a reasonable instrument for behavioural
   questions.
2. **The causal audit does not transfer.** The surrogate's CDVA seed score alone reaches
   AUC 0.54 to 0.63, near chance on three of four pairs.
3. **It adds nothing on top of behaviour.** Incremental AUC is between -0.000 and +0.004.
   The surrogate's causal label carries no information about the API model beyond what its
   behaviour already carries.

Point 3 holds even in the same-family case. Qwen2.5-7B is the closest available proxy for
Qwen3-Next-80B, and its causal audit still adds 0.004 AUC.

## 3. What this means for the TIST manuscript

This is a negative result and it should be reported as one. It is also the strongest
policy sentence available to the paper: a causal validity audit cannot be outsourced to an
open stand-in, not even a same-family stand-in, so an auditing regime that accepts
black-box access accepts a behavioural score with no causal warrant. That connects the
NIST AI RMF and EU AI Act framing in the introduction to a concrete requirement, which is
the kind of deployment-facing argument TIST readers expect.

Placement: Discussion, not Results, and it must not be sold as a method.
