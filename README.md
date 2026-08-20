# MIRAGE: Auditing the Data Quality and Measurement Validity of LM Bias Benchmarks

A bias benchmark is a dataset, and its score inherits every quality defect of the items that
produced it. MIRAGE audits those items one at a time. It filters malformed items, tests
whether a model's answer survives edits that should not change it, and measures how far the
answer moves when the internal representation of the protected attribute is replaced. What it
returns is a validity report for the benchmark's items, conditioned on the models audited —
not another score for the model.

Everything lives in [`Code/audit`](Code/audit). This repository is the artifact for the
accompanying paper: the audit toolkit, the released probe sets, every results table, and a
per-number provenance file that maps each figure in the paper back to the file that produced
it.

---

## 0. Honest headline

1. **The first finding is about the data, not the models.** Malformed counterfactual
   substitutions — where a swap lands on text that is already an answer option, so the item
   offers the same option twice — account for **1,177 of the 5,960** counterfactual pairs the
   unfiltered expansion yields per model (**19.7%**). Mean intervention magnitude is higher on
   those malformed pairs than on the cleaned pool for **all four** open-weight models, so the
   contamination runs in one direction rather than averaging out.

2. **The measurement is specific, and the controls say so.** The protected-attribute
   intervention produces mean `|C|` **1.51–2.49×** that of a matched content-token control and
   **1.99–3.36×** that of a function-word control, on every open-weight model, with every
   cluster-bootstrap interval above 1. A third control that injects an activation from an
   unrelated seed does **not** separate — it is a larger perturbation, not a null condition,
   and it is reported and excluded from the placebo claim rather than quietly dropped.

3. **Analyst choices move the answer.** Switching the threshold convention from the percentile
   rule to the control-calibrated one reverses the ranking of Qwen2.5-7B and Gemma-2-2B. An
   absolute validity rate reported without the convention that produced it is not
   interpretable.

4. **A competence gate is not optional for translated probes.** Gemma-2-2B locates the
   counterfactual position in *every* Bengali probe — which reads as full coverage — while
   answering the Bengali competence probe at 0.221, below the 0.333 chance rate. The gate
   admits **13 of 16** model-language pairs and keeps a comprehension failure from being
   scored as bias.

5. **Intervention verdicts do not transfer.** Across four open-to-closed pairings, adding an
   open stand-in's intervention verdict to a behavioural predictor changes AUC by between
   **−0.0001 and +0.0037** — including for the pairing that shares a training lineage.
   Black-box access supports behavioural auditing only.

Nothing here is tuned to look good. Phi-4-mini's rates are reported and then explicitly
*declined* as a bias reading, because it fails the unmodified benchmark item.

---

## 1. The probe sets

Derived from three source benchmarks and released as a dataset in their own right. Counts are
after the integrity filter removed 297 malformed slot-(c) variants, spread over 286 of the 596
English seeds; no seed dropped below the minimum of three surviving variants, so the removal
cost no seeds.

| Set | Seeds | Prompts | Protected axes |
|---|---:|---:|---:|
| BBQ | 254 | 2,922 | 18 |
| CrowS-Pairs | 181 | 2,003 | 12 |
| StereoSet | 161 | 1,930 | 4 |
| **English total** | **596** | **6,855** | **22** |
| Hindi (translated from BBQ) | 230 | 2,644 | 9 |
| Bengali (translated from BBQ) | 222 | 2,548 | 9 |
| Synthetic controls | 200 | 1,200 | 5 |

Each seed expands into a **pentad** — twelve prompts over five slots. Slot (a) is the surface
item; (b) neutralises the protected token; (c) substitutes five admissible values on the same
axis and supplies the counterfactual pairs; (d) adds one valid and one misleading context
sentence; (e) adds neutral, stereotype-pushing and anti-stereotype-pushing frames. Slots (a)
to (c) are template-deterministic (3,875 of the 6,855 English prompts); slots (d) and (e) are
drafted by DeepSeek-Chat and validated by two human annotators against
[`annotation_guidelines.md`](Code/audit/results/tist/e3/annotation_guidelines.md).

**Models.** Four open-weight, which get both stages: Llama-3.1-8B and Gemma-2-2B via
TransformerLens, Qwen2.5-7B and Phi-4-mini via NNsight. Four API-served, which get the
behavioural stage only (the intervention needs the residual stream): Mistral-Medium,
Qwen3-Next-80B, Nova-2-Lite, Gemini-2.5-Flash. 81,056 behavioural responses in total.

---

## 2. The measurement

### 2.1 The commutator

Take two variants of one seed differing only in the protected term, `a` and `b`. Run both.
Cache the residual stream of the `a` run at the protected-term position, write it into the `b`
run at the same position at every decoder layer, leave every other position untouched, and
read the change in the gold-answer logit:

```
C(a, b) = logit_gold( swap(a -> b) ) - logit_gold( b )
```

Implemented in [`GPU_CPU/cdva_patching.py`](Code/audit/GPU_CPU/cdva_patching.py) and
[`GPU_CPU/utils_attention.py`](Code/audit/GPU_CPU/utils_attention.py). `C ≈ 0` means the answer
does not depend on that representation. A large `|C|` is evidence of sensitivity **to this
intervention** — not proof that the same pathway runs during ordinary inference, and not a
claim about real-world discrimination.

### 2.2 The threshold

`tau = 0.456`, shared by all four open-weight models. It is the median of the per-model
Youden-optimal cuts (0.376–0.499) on the 200 synthetic control seeds, so the threshold is tied
to the audit's own criterion rather than to the shape of the observed distribution.
Split-half validation on those same controls puts optimism at 0.027 at worst
([`seed_level/threshold_validation.csv`](Code/audit/results/tist/seed_level/threshold_validation.csv)).

The superseded percentile convention (75th percentile of `|C|`) puts the effective cut at
`1.178`. Both are reported, because the choice between them reorders two models.

### 2.3 The two rates

- **MIRAGE-B**, the native pass rate: the fraction of seeds passing all five slot checks —
  gold on (a), gold on (b), majority-stable over distinct (c) substitutions, gold on both (d)
  variants, majority-robust over the three (e) frames. See
  [`CPU_Only/scoring.py`](Code/audit/CPU_Only/scoring.py)`::compute_mirage_b`.
- **MIRAGE-Full**, the audit-robust pass rate: MIRAGE-B **and** the seed clears the
  intervention gate.
- **Validity gap**: MIRAGE-B minus MIRAGE-Full. It states how far a native score exceeds what
  survives scrutiny — a claim about the evidence, not about the fairness of the model.

> ### ⚠ Two per-seed reduction rules are in play. Do not mix them.
>
> A seed has many counterfactual pairs (median 20), and they are reduced to one verdict two
> different ways in this codebase:
>
> | Rule | Where | Llama-3.1-8B MIRAGE-Full |
> |---|---|---:|
> | **max** over pairs — every pair must clear `tau` | `TIST/failure_analysis.py`, `TIST/tau_validation.py` → `seed_level/*.csv` | **0.069** |
> | **mean** over pairs — the seed's mean `\|C\|` must clear `tau` | `TIST/e2b_calibrated_tau.py` → `e4/stats_e2b_calibrated.csv` | **0.196** |
>
> The **max** rule is the paper's primary definition and backs every per-source and
> decomposition number. The **mean** rule appears only in the threshold-convention comparison,
> where the two threshold columns are meant to be read against each other. Comparing a rate
> from one against a rate from the other is an error, not a discrepancy in the data.

---

## 3. Results of the completed run

Pooled over the 596 English seeds. Rates use the max reduction rule.

| Model | MIRAGE-B | Intervention gate | MIRAGE-Full | Control AUC |
|---|---:|---:|---:|---:|
| Llama-3.1-8B | 0.250 | 0.223 | **0.069** | 0.727 |
| Gemma-2-2B | 0.106 | 0.379 | **0.032** | 0.997 |
| Qwen2.5-7B | 0.176 | 0.025 | **0.002** | 0.830 |
| Phi-4-mini | 0.000 | 0.074 | **0.000** | 0.933 |

**Phi-4-mini is not a bias result.** It returns the gold answer on slot (a) — the unmodified
benchmark item — on only 0.101 of seeds, and 68.8% of its responses had to be recovered by the
deterministic fallback parser rather than read from the requested format. Its rates measure
instruction-following, and they are excluded from every ranking claim.

**Per source**, the low rates are not a BBQ artefact: they occur on CrowS-Pairs too (Llama
0.083, Gemma 0.011), and StereoSet yields at most 0.006 on any model — uniform enough that it
reads as a property of the item form, not of the models.

**Failure is distributed, not concentrated.** Of the 555 seeds Llama-3.1-8B fails, 64.0% fail
both stages, 19.5% pass all five slot checks but exceed `tau` on at least one pair, and 16.6%
do the reverse. Neither stage manufactures the result alone.

**Wall clock**: 4.55 hours total for the four open-weight models on one A100-SXM4-80GB
([`environment.json`](Code/audit/results/tist/environment.json)).

---

## 4. Reproducing

```bash
cd Code/audit
cp .env.example .env          # fill in the keys you need; .env is git-ignored

# 1. Build the probe sets (slots a/b/c are deterministic; d/e call the generator)
python run_dataset.py                 # add --det-only to skip the generator
python run_dataset.py --force         # regenerate even if cached

# 2. Behavioural evaluation of the API-served models (CPU, checkpoints every 50 prompts)
python Dry_Run/dry_run_cpu_only.py    # dry run first
python run_cpu_full.py

# 3. The GPU battery: integrity, placebo/layer/direction/control/mediation, multilingual,
#    surrogate transfer
python TIST/run_tist_gpu.py --tasks all --dry      # 2 units per task, real code path
python TIST/run_tist_gpu.py --tasks all
python TIST/run_tist_gpu.py --tasks e1_battery --models llama-3.1-8b-instruct

# 4. Seed-level statistics, threshold validation, failure decomposition, tables
python TIST/seed_level.py
python TIST/tau_validation.py
python TIST/failure_analysis.py
python TIST/e2b_calibrated_tau.py
python TIST/make_tables.py
```

Every stage checkpoints to disk before the next begins, so an interrupted run resumes without
repeating completed work. `TIST/deploy_tist.py` provisions a cloud GPU lease and
`TIST/bootstrap_tist.sh` is the remote entrypoint.

---

## 5. Where each number comes from

[`results/tist/number_provenance.csv`](Code/audit/results/tist/number_provenance.csv) maps
every table and figure in the paper to the file that produced it.
[`seed_level/provenance_audit.csv`](Code/audit/results/tist/seed_level/provenance_audit.csv)
re-derives each headline count from source and records whether it matched. Both are
regenerated by `TIST/make_tables.py`, so they cannot drift from the tables.

```
Code/audit/
  config.py               models, datasets, tau, API routing, .env contract
  run_dataset.py          probe-set construction (single entry point)
  run_cpu_full.py         API behavioural evaluation
  Dataset/                seed sampling, pentad generation, translation, validation
  GPU_CPU/                model loading, behavioural eval, activation patching
  CPU_Only/               scoring (MIRAGE-B / MIRAGE-Full), statistics, leaderboard
  GPU_Remaining/          GPU-only instrument checks (patch-site recovery, temperature,
                          option-order) reusing the same model and patching code
  TIST/                   the experiment battery, seed-level statistics, table generation
  Dry_Run/                dry runs that exercise the real code path on 2 units per task
  results/                behavioural, commutator and scored results
    tist/e0/              item integrity
    tist/e1/              placebo, layer sweep, direction, controls, mediation, metrics
    tist/e2/              threshold sweep and calibration
    tist/e3/              annotation sheets, guidelines, inter-annotator agreement
    tist/e4/              multilingual probes, competence gate, calibrated rates
    tist/e5/              open-to-closed surrogate transfer
    tist/seed_level/      the seed-clustered statistics the paper reports
    tist/tables/          generated LaTeX tables
```

---

## 6. Statistical protocol

The **seed** is the inferential unit throughout. A seed's twelve prompts and twenty-odd
counterfactual pairs share a passage, an answer set and a protected axis, so treating them as
independent would understate every standard error. Quantities are aggregated to one value per
seed before testing; confidence intervals come from a cluster bootstrap that resamples seeds
and carries each seed's pairs with it; families of related tests carry Holm correction. Testing
at the level of the derived pair returns p-values that describe the number of pairs rather than
the strength of the effect.

---

## 7. Environment and secrets

Every key is read from the environment. No secret is written into a tracked file. Copy
`Code/audit/.env.example` to `Code/audit/.env` and fill what you need — the `.env` is
git-ignored.

| Variable | Purpose |
|---|---|
| `HUGGINGFACE_TOKEN` | Model download |
| `DEEPSEEK_API_KEY_1..2` | Slot (d)/(e) generation and the default JSON-repair judge |
| `OPENROUTER_API_KEY_1..2` | Fallback route for the API-served models |
| `GEMINI_API_KEY_1..4` | Optional JSON-repair judge and generation fallback |
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Bedrock route for Qwen3-Next-80B and Nova-2-Lite |
| `MISTRAL_API_KEY1..2` | Mistral-Medium |
| `MEGALLM_API_Key`, `GeminiCheap_LinkAPI_Key` | Gateway routes for Gemini-2.5-Flash |
| `Github_Classic_Token` | Checkpoint pushes from a remote GPU |

Neither DeepSeek nor Gemini is an evaluation model. They appear only as the slot (d)/(e)
generator and as the JSON-repair route for responses the deterministic parser cannot read;
`results/tist/e3/` and `tab_repair` report how often each route was used.

A single 40 GB+ GPU is enough. `Code/audit/requirements.txt` and
`Code/audit/GPU_Remaining/requirements_gpu.txt` pin the environment;
`results/tist/environment.json` records the exact versions of the completed run.

---

## 8. Citations

The intervention instrument is activation patching (Meng et al. 2022, arXiv:2202.05262; Zhang
and Nanda 2024, arXiv:2309.16042), applied here as an item-level validity measurement rather
than as a study of model mechanism. The causal-mediation comparison follows Vig et al. 2020
(arXiv:2004.12265). The construct-validity vocabulary comes from Cronbach and Meehl (1955),
Messick (1995) and Jacobs and Wallach (2021, arXiv:1912.05511). The audited benchmarks are BBQ
(Parrish et al. 2022, arXiv:2110.08193), CrowS-Pairs (Nangia et al. 2020, arXiv:2010.00133) and
StereoSet (Nadeem et al. 2021, arXiv:2004.09456); the multilingual probes follow MBBQ
(Neplenbroek et al. 2024, arXiv:2406.07243). Patching runs through TransformerLens (Nanda and
Bloom 2022) and NNsight (Fiotto-Kaufman et al. 2025, arXiv:2407.14561).

The probe sets contain stereotyped statements about protected groups. That is what makes them
usable for measurement, and the dataset card says so before anything else.
