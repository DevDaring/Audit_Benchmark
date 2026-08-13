# TIST resubmission — status

Updated at the end of Phase 1 (CPU work, code, deployment assets).

---

## Completed

| Item | Outcome |
|---|---|
| Phase 0, environment | Akash reachable on `Bharat_AKASH_Key` (0 active deployments). `ssh debkoushik` up: 32 cores, 31 GB RAM, no GPU. MiKTeX present locally, so a LaTeX build gate is achievable. No torch locally, so all model work is remote. |
| E2, threshold | Sweep and scale audit complete. **Found a unit mismatch in the published tau.** |
| E5, API surrogate | Complete. Causal labels do not transfer to closed models. |
| E0, item integrity | New, unrequested. **Found 297 malformed slot-c counterfactuals** carrying 19.75% of CDVA pairs. |
| E1.4, controls | 200 synthetic control seeds built and structurally validated, 1,200 prompts, 5 axes. |
| GPU battery code | `TIST/patch_core.py` and `TIST/run_tist_gpu.py` cover E1.1 to E1.6 and E4, with JSONL resume. |
| Deployment | `TIST/sdl_tist.yaml`, `TIST/bootstrap_tist.sh`, `TIST/deploy_tist.py` ready for a 24 GB lease. |

## Running

| Item | Progress |
|---|---|
| E4 translation, Hindi and Bengali | ~100 of 508 seed x language units. About 10% fail verbatim validation and are retried on the next pass. |

## Three defects found in the existing pipeline

All three change numbers in the rejected manuscript. Full write-ups in `e0/FINDINGS.md`
and `e2/FINDINGS.md`.

1. **tau unit mismatch.** tau was computed as the 75th percentile of `|delta_logit|` in
   logit units, then compared against `cdva_seed_score`, a [0,1] invariance score. The
   effective cut sits at the 85.4th percentile, not the 75th. Correcting it lowers
   MIRAGE-Full on three of four models, Qwen2.5-7B by 34% relative.
2. **Degenerate counterfactuals.** 297 of 2,980 slot-c variants substitute a term that is
   already another answer option, producing items with two identical options. They carry
   19.75% of CDVA pairs and show 44% higher mean |C| on Qwen and 47% higher on Llama.
   Excluding them costs no seeds.
3. **Ordering-stability claim too strong.** Section 6.8 of the rejected paper said model
   ordering is stable under threshold choice. Llama first and Phi last are stable, but
   Gemma and Qwen swap at the 80th percentile, and the published tau sat past the swap.

Defect 2 is the strongest available answer to Reviewer 1: part of the commutator
demonstrably was an item-construction artefact, and the audit's own bookkeeping found it.

## Next

1. Push code so the lease can clone it.
2. Provision the 24 GB Akash lease, run the battery across four models.
3. E3 annotation protocol and sheets, while the GPU runs.
4. E2b, calibrate tau on the E1.4 controls with Youden's J.
5. Analysis, figures, tables.
6. Manuscript and cover letter.
7. Verification gates.

## Estimated GPU cost

About 52 GPU-hours, derived from measured per-pair patching cost of 0.17 to 0.58 s and
the measured 22.2 GPU-hours of the original behavioural run. Approved on a 24 GB
RTX-class lease.
