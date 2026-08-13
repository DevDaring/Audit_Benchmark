# E2 — threshold calibration and sensitivity

Answers Reviewer 2, point 2. Produced by `Code/audit/TIST/e2_tau_sensitivity.py`
from `results/cdva_results.parquet` (23,840 pairs, all success_flag=True, no position
fallbacks) and `results/scored_results.parquet` (596 seeds x 4 open models).
No model was run. Artifacts: `scale_audit.json`, `tau_sweep.parquet`,
`stats_tau_sensitivity.csv`, `stats_tau_rank_table.csv`, `stats_tau_ordering.csv`.

---

## 1. The published threshold carries a unit mismatch

`run_cpu_postprocess.py:33` sets tau to the 75th percentile of `|delta_logit|`, which is
0.7644 **in logit units**. `CPU_Only/scoring.py:171` then compares that value against
`cdva_seed_score`, which `GPU_CPU/cdva_patching.py:59` defines as
`1 - min(|delta_logit| / 5, 1)`, a **[0,1] invariance score**. The two live on different
scales. The comparison only type-checks because 0.7644 logits happens to land inside the
unit interval.

`GPU_CPU/cdva_calibration.py:26` searches candidate thresholds on `linspace(0.1, 0.9, 17)`,
which confirms tau was designed as a score threshold. The percentile fallback was written
on the wrong scale.

**Effect.** Comparing a score of `1 - |C|/5` against 0.7644 imposes a cut at
`|C| = 5 x (1 - 0.7644) = 1.178` logits. That is the **85.4th percentile** of |C|, not the
75th the manuscript states. The rejected paper therefore reported a more permissive
threshold than it claimed.

The scale-consistent threshold for a genuine 75th percentile is `tau = 1 - 0.7644/5 = 0.8471`.

| Model | MIRAGE-Full, published tau | MIRAGE-Full, scale-corrected 75th pct | Change |
|---|---|---|---|
| Llama-3.1-8B-Instruct | 0.2433 | 0.2248 | -0.0185 |
| Qwen2.5-7B-Instruct | 0.1191 | 0.0789 | -0.0403 (-34% relative) |
| Gemma-2-2B-it | 0.0973 | 0.0940 | -0.0034 |
| Phi-4-mini-Instruct | 0.0000 | 0.0000 | 0.0000 |

Integrity check: recomputing with the published tau reproduces every stored
`mirage_full_pass` label exactly, so the artifacts are internally consistent and the
defect is confined to how tau was derived.

## 2. Sensitivity sweep, 10th to 90th percentile of |C|

`stats_tau_sensitivity.csv` carries MIRAGE-Full per model at each percentile with 95%
bootstrap CIs (5,000 resamples over the 596 seeds, seed 20260101). Absolute rates move a
great deal: Llama runs from 0.002 at the 10th percentile to 0.250 at the 90th, Qwen from
0.000 to 0.139. Reviewer 2's concern about absolute pass rates is correct and the paper
must state the threshold with its units every time it reports a MIRAGE-Full number.

## 3. Ordering is stable at the ends and unstable in the middle

Llama ranks first and Phi-4-mini ranks last at every threshold in the sweep. The two
middle models swap:

- percentiles 30 to 75: Llama > Gemma > Qwen > Phi
- percentiles 80 to 90: Llama > Qwen > Gemma > Phi

The published tau sits at an effective 85.4th percentile, that is, on the far side of the
swap. The Qwen-above-Gemma ordering the rejected manuscript reported is an artefact of the
unit mismatch. Under a scale-consistent 75th-percentile rule the ordering is Gemma above
Qwen.

Kendall tau_b against the published ordering is 0.55 to 0.71 through the middle of the
sweep and 1.00 from the 80th percentile up (`stats_tau_ordering.csv`). With four models
the statistic has essentially no power, and no p-value in the table is significant, so the
rank table above is the evidence the paper should show, not the correlation.

The rejected manuscript's Section 6.8 claim that model ordering is stable under threshold
choice is therefore **too strong** and must be restated: the extreme ranks are
threshold-invariant, the middle two are not.

## 4. What this means for the TIST manuscript

1. Report the corrected, unit-consistent threshold and state its units.
2. Recompute every MIRAGE-Full number under it. Three of four models move.
3. Replace the "ordering is stable" sentence with the rank table and the swap point.
4. Keep the percentile rule only as a fallback. E2b will calibrate tau on the E1.4
   synthetic controls with Youden's J, which gives a threshold with an external criterion
   instead of a distributional convention. That is the substantive answer to Reviewer 2.

## 5. Open item

E2b (Youden calibration against ground-truth controls) is blocked on E1.4, which needs GPU.
