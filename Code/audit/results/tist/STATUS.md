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
| GPU battery, Akash A100 80 GB | Lease 4 (dseq 1786608581463). Longest-job-first, three models concurrent. |

## Lease history, and what each failure taught

Four leases were needed. Each failure produced a permanent guard, so the record is kept
here rather than tidied away; three of the four faults were in this repository's own code.

| # | GPU | Outcome | Fault | Guard added |
|---|---|---|---|---|
| 1 | RTX 4090 | reported COMPLETE having computed nothing | `attn_implementation="flash_attention_2"` requested unconditionally; transformers raises rather than degrading, so all four models failed to load. `run_tist_gpu` caught the failures and still exited 0. | sdpa fallback in `load_osm.load_model`; exit code 3 when no model loads; dry-run gate requires at least one output record, not just rc=0 |
| 2 | A100 80 GB | dry run produced no records, container held for inspection | `task_e4_behav` called `evaluate_osm_model` with the wrong signature, crashing before any patching task ran | signature corrected; `TIST/check_signatures.py` verifies every production call site without needing torch; tasks isolated in try/except; patching ordered before the behavioural pass |
| 3 | RTX 4090 | closed deliberately | 24 GB is too tight for Llama-3.1-8B under TransformerLens, which holds a converted model beside the HF weights | 40 GB VRAM floor in `deploy_tist.py`, refusing under-spec cards passed through `--gpus` |
| 4 | A100 80 GB | closed deliberately after 40 min | `push_results` ran `git pull --rebase` with output discarded; one conflict left the repo mid-rebase and wedged every later checkpoint invisibly, including the final COMPLETE push | rebase aborted before and after each attempt, push retried three times, outcome reported; standalone `TIST/sync_results.sh` daemon |

Two performance faults were found alongside them. The pinned torch 2.5.1 did not survive
dependency resolution and the container runs torch 2.13.0+cu130, so the hard-coded
flash-attn wheel could not install; the wheel URL is now derived from the torch that
actually installed and the run proceeds on sdpa when none matches. The CUDA image ships
without Python headers, so Triton could not compile and batched generation silently fell
back to single-prompt decoding; `python3-dev` is now installed.

Scheduling was also wrong. Models were dispatched largest-VRAM-first, which put
Phi-4-mini last. Phi decodes at 4.54 s/prompt against Llama's 0.99, so it is the longest
job at about 12 h against 4.8 h despite being the smallest model, and starting it last set
the finish time. Dispatch is now longest-job-first.

**Reproducibility note for the manuscript.** The battery runs on torch 2.13.0+cu130 with
sdpa attention, while the original CDVA results were produced on torch 2.5.1+cu124. The E1
comparisons are internally consistent, since every condition runs on the same stack, but
E1.4 and E4 are compared against stored English values and that comparison needs the
cross-stack consistency check listed under Next.

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

## Action required from the authors

`e3/iaa_sheet_koushik.csv` and `e3/iaa_sheet_abhinaba.csv` are ready to fill. 200 items,
stratified over 61 strata: 87 slot (d), 113 slot (e). Criteria and worked examples are in
`e3/annotation_guidelines.md`. Roughly 60 to 90 minutes each, independently, no conferring
until both are done. Then:

```
python TIST/e3_annotation.py score
```

That emits Cohen's kappa between the two of you, Fleiss' kappa across the LLM panel, and
the agreement of the LLM majority with your consensus. Without these two files the paper
reports the LLM panel only, which is a weaker answer to Reviewer 2.

## Next

1. Finish the Hindi and Bengali pentads, commit the parquets.
2. Provision the 24 GB Akash lease and run the battery across four models:
   `python TIST/deploy_tist.py --deposit 20 --wait 180`
   Progress: `python TIST/deploy_tist.py --status`. Results push every 15 minutes.
   Close the lease when `results/tist/TIST_GPU_DONE` appears:
   `python TIST/deploy_tist.py --close`
3. `python TIST/e1_analyse.py` and `python TIST/e4_analyse.py` once results land.
4. API-model multilingual runs on `debkoushik` under tmux.
5. E2b, adopt the Youden threshold from E1.4 and recompute every MIRAGE-Full number.
6. Figures F-A to F-E, script-generated tables.
7. Manuscript and cover letter.
8. Verification gates.

## Cost control

The lease bills continuously. Close it as soon as `TIST_GPU_DONE` appears; the estimate is
about 40 GPU-hours of work plus setup, so budget roughly two days of wall clock and check
in at least twice a day.

## Measured throughput, for the deployability table

TIST asks for runtime and cost figures, so these are measured rather than estimated. All
from lease 5, one A100 80 GB, sequential, sdpa attention, per-seed memoisation enabled.

| Sync checkpoint (UTC) | Records | Interval rate |
|---|---|---|
| 10:03:16 | 2,315 | |
| 10:18:18 | 4,761 | 163 / min |
| 10:33:20 | 7,249 | 165 / min |

Stable at roughly **164 records per minute**, where a record is one unit of audit work:
a counterfactual pair for the battery and the controls, a layer window for the sweep, a
pair for the multilingual CDVA.

Total work is about 121,400 units across the four models, so the patching battery costs
roughly **12 GPU-hours** and the multilingual behavioural pass a further **4**, about
**16 GPU-hours** for the whole audit, near $60 at the A100 rate this lease pays.

Two caveats the manuscript must carry. The run uses sdpa rather than flash-attention,
because the pinned torch did not survive dependency resolution in the container and no
matching wheel exists for the version that installed; numerics are unaffected but the
figures are conservative. And the throughput is single-process: this provider allocates
the GPU as one indivisible unit and kills a second process attaching to it, so the
figures describe one model at a time rather than a saturated card.

An earlier measurement of 1,527 records per minute is not comparable and must not be
quoted: it came from a two-process configuration that this provider terminates.

## Estimated GPU cost

About 52 GPU-hours, derived from measured per-pair patching cost of 0.17 to 0.58 s and
the measured 22.2 GPU-hours of the original behavioural run. Approved on a 24 GB
RTX-class lease.
