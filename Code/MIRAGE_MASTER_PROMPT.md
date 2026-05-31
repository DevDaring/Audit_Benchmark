# MIRAGE — Master Prompt for Codebase Generation

This document is the complete specification for building the MIRAGE research codebase. Feed this entire document to a coding agent (Claude Code, Cursor, Copilot, Codex, etc.). The agent must produce a runnable repository that satisfies every requirement below.

---

## 0. Research Context

MIRAGE (Mechanism-Indexed Reliability Audit for Group-bias Evaluation) is a discriminative-validity audit framework for LLM bias benchmarks. It operationalizes the Epistematics methodology (Kalaitzidis 2026, "The Evaluation Trap") for bias measurement by combining behavioral probing with causal activation patching. Target venue is an SCIE-indexed journal.

Key prior works the codebase must cite in comments:

- Kalaitzidis (2026). "The Evaluation Trap: Benchmark Design as Theoretical Commitment." arXiv:2605.14167 — Epistematics framework, 5 failure modes (FM1 proxy substitution, FM2 architectural indistinguishability, FM3 context blindness, FM4 criterion leakage, FM5 approximation ceiling).
- Wang et al. (2025). "Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs." ACL 2025.
- Bean et al. (2025). "Measuring what Matters: Construct Validity in Large Language Model Benchmarks." NeurIPS 2025 Datasets and Benchmarks Track.
- Parrish et al. (2022). "BBQ: A Hand-Built Bias Benchmark for Question Answering." Findings of ACL 2022.
- Nangia et al. (2020). "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models." EMNLP 2020.
- Nadeem et al. (2021). "StereoSet: Measuring stereotypical bias in pretrained language models." ACL-IJCNLP 2021.
- Zhao et al. (2018). "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods." NAACL 2018 (WinoBias).
- Shaikh et al. (2023). "On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning." ACL 2023.
- Liu et al. (2026). "DIFFHEADS: Differential Head Analysis for Bias in LLMs." AAAI 2026.
- Pearl, J. (2009). *Causality*. Cambridge University Press — for do-calculus / interventional framing.

Every code file must carry a header comment block listing the papers it implements, builds on, or cites. Citation hygiene is mandatory because these comments feed directly into the paper's references.

---

## 1. Repository Layout

```
mirage/
├── README.md                    # One-stop documentation
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Template (no real keys)
├── .gitignore                   # Must ignore .env, results/, cache/
│
├── Dry_Run/                     # Sanity checks, one file per other folder
│   ├── dry_run_dataset.py
│   ├── dry_run_gpu_cpu.py
│   ├── dry_run_cpu_only.py
│   └── dry_run_all.py           # Master dry run that calls the three above
│
├── Dataset/                     # Dataset download, format validation, pentad generation
│   ├── download_bbq.py
│   ├── download_crows_pairs.py
│   ├── download_stereoset.py
│   ├── download_winobias.py
│   ├── sample_seeds.py          # Stratified seed selection with fixed RNG seed
│   ├── pentad_generator.py      # Builds all 5 probe slots per seed
│   ├── cot_attack_generator.py  # Calls DeepSeek API for slot (e)
│   ├── context_shift_drafter.py # Calls DeepSeek API for slot (d)
│   ├── validate_pentad.py       # Schema + completeness validation
│   └── seeds/                   # Stored seed dataset (parquet)
│
├── GPU_CPU/                     # OSM evaluation + CDVA causal patching
│   ├── load_osm.py              # bf16 + flash-attention loading
│   ├── osm_behavioral.py        # Pentad evaluation across 4 OSM models
│   ├── cdva_patching.py         # Causal activation patching
│   ├── cdva_calibration.py      # τ threshold calibration on dev split
│   └── utils_attention.py       # TransformerLens / nnsight helpers
│
├── CPU_Only/                    # API evaluation + analysis
│   ├── api_clients/
│   │   ├── bedrock_client.py    # AWS Bedrock with OpenRouter fallback
│   │   ├── gemini_client.py     # GCP Gemini round-robin
│   │   ├── mistral_client.py    # Mistral platform round-robin
│   │   └── openrouter_client.py # Round-robin OpenRouter fallback
│   ├── api_behavioral.py        # Pentad evaluation across 4 API models
│   ├── judge_router.py          # Optional judge / answer-extraction
│   ├── scoring.py               # MIRAGE-B, MIRAGE-Full composite scores
│   ├── statistics.py            # Bootstrap CIs, McNemar, Cohen's h
│   ├── leaderboard.py           # Per-benchmark validity vectors (FM1-FM5)
│   ├── predictive_validity.py   # Held-out WinoBias + HolisticBias check
│   └── results_analysis.py      # Final tables and figures
│
└── results/                     # All output here (gitignored)
    ├── pentad_dataset.parquet
    ├── behavioral_results.parquet
    ├── cdva_results.parquet
    ├── scored_results.parquet
    ├── leaderboard.parquet
    └── figures/
```

---

## 2. Environment

- OS: Ubuntu 22.04 / 24.04 LTS, x86_64. Flash-attention does not support Windows or macOS — error out clearly if those are detected.
- Python: 3.12 (cp312 wheels required for flash-attention).
- GPU: single NVIDIA L4 24GB or equivalent (CUDA 12.4).
- **No virtual environments.** Use the global Python install directly. Do not call `python -m venv`, `virtualenv`, `conda`, or `poetry`.

### 2.1 Install commands (must appear verbatim in README)

```
python3 -m pip install --upgrade pip setuptools wheel \
  && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  && python3 -m pip install numpy<2.0 transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
       bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
       sentencepiece==0.2.0 protobuf==4.25.0 \
  && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
  && python3 -m pip install --no-deps /tmp/flash_attn.whl
```

### 2.2 Additional pinned packages

Append to requirements.txt:
- `transformer_lens==2.9.0` (for Llama and Gemma activation patching)
- `nnsight==0.3.7` (for Qwen and Phi-4 activation patching)
- `boto3>=1.34` (AWS Bedrock)
- `google-generativeai>=0.8` (Gemini)
- `mistralai>=1.2` (Mistral)
- `openai>=1.50` (OpenRouter is OpenAI-compatible)
- `outlines==0.1.0` (constrained JSON decoding for OSM)
- `pyarrow>=15` (parquet I/O)
- `scipy>=1.14`, `statsmodels>=0.14` (statistics)
- `scikit-learn>=1.5` (predictive validity classifier)
- `matplotlib>=3.9`, `seaborn>=0.13` (figures)

Verification block (run after install):

```
python3 -c "
import torch, bitsandbytes as bnb, importlib
print('torch:', torch.__version__, '| cuda:', torch.version.cuda)
print('bitsandbytes:', bnb.__version__)
fa = importlib.import_module('flash_attn')
print('flash-attn:', getattr(fa, '__version__', 'unknown'))
print('ALL INSTALLED')
"
```

---

## 3. Environment Variables (.env file)

All keys MUST be loaded via `python-dotenv` from `.env` at the repository root. **No hardcoded keys anywhere — including test files.** The agent must scan all generated files for accidental literal keys before completing.

Required variables:

| Variable | Purpose |
|---|---|
| `HF_KEY` | HuggingFace model downloads |
| `AWS_BEDROCK_KEY` | AWS Bedrock primary access (gpt-oss-20b, Amazon Nova 2 Lite) |
| `OPENROUTER_API_KEY_1` | OpenRouter fallback, round-robin slot 1 |
| `OPENROUTER_API_KEY_2` | OpenRouter fallback, round-robin slot 2 |
| `GCP_Key1` | Gemini API key 1 (round-robin) |
| `GCP_Key2` | Gemini API key 2 (round-robin) |
| `GCP_Key3` | Gemini API key 3 (round-robin) |
| `GCP_key4` | Gemini API key 4 (round-robin) — note lowercase 'k' is intentional |
| `MISTRAL_API_KEY1` | Mistral Medium / Small key 1 (round-robin) |
| `MISTRAL_API_KEY2` | Mistral Medium / Small key 2 (round-robin) |
| `DEEPSEEK_API_KEY_1` | DeepSeek key 1 (round-robin) — used for generation and judge |
| `DEEPSEEK_API_KEY_2` | DeepSeek key 2 (round-robin) |

A `.env.example` file containing every key with empty values must exist in the repo.

`.gitignore` must include: `.env`, `results/`, `cache/`, `__pycache__/`, `*.pyc`.

---

## 4. Model Stack

### 4.1 OSM models (4) — evaluated with full audit (behavioral + CDVA)

| Slot | HuggingFace ID | Library for patching |
|---|---|---|
| OSM-1 | `meta-llama/Llama-3.1-8B-Instruct` | TransformerLens |
| OSM-2 | `Qwen/Qwen2.5-7B-Instruct` | nnsight |
| OSM-3 | `google/gemma-2-2b-it` | TransformerLens |
| OSM-4 | `microsoft/Phi-4-mini-instruct` | nnsight |

All loaded in bf16 with flash-attention-2 enabled. Verify each loads on a single L4 24GB before main runs.

### 4.2 Evaluation APIs (4) — behavioral audit only

| Slot | Model ID | Primary route | Fallback route |
|---|---|---|---|
| API-1 | `openai.gpt-oss-20b-1:0` | AWS Bedrock via `AWS_BEDROCK_KEY` | OpenRouter (`OPENROUTER_API_KEY_1` / `_2` round-robin) |
| API-2 | `amazon.nova-2-lite-v1:0` | AWS Bedrock via `AWS_BEDROCK_KEY` | OpenRouter (round-robin) |
| API-3 | `gemini-3-flash-preview` | GCP via `GCP_Key1`..`GCP_key4` round-robin | None — skip on persistent failure |
| API-4 | `mistral-medium-3.5` | Mistral platform via `MISTRAL_API_KEY1` / `2` round-robin | None — skip on persistent failure |

**Retry & fallback rules (apply identically to API-1 and API-2):**
1. Attempt primary route (Bedrock). If failure, retry once with same key.
2. If still failing, switch to OpenRouter. Round-robin between `OPENROUTER_API_KEY_1` and `OPENROUTER_API_KEY_2`, two attempts total.
3. If all four attempts fail, **skip the row** but record the failure with a flag. Do not raise.

**For API-3 and API-4:**
1. Round-robin within the provider's available keys, max 2 attempts per key.
2. If exhausted, skip and flag.

Every API call records: attempt count, route used (bedrock / openrouter / gcp / mistral), key index used, latency, and final status flag.

### 4.3 Generator (template generation only — never evaluated)

| Use | Model | Keys |
|---|---|---|
| CoT-attack prompt generation, context-shift drafting, any other template generation | DeepSeek Chat via DeepSeek platform | `DEEPSEEK_API_KEY_1`, `DEEPSEEK_API_KEY_2` round-robin |

DeepSeek MUST NOT appear anywhere in the evaluation pipeline. All template generation outputs are logged with the generator's model version and timestamp.

### 4.4 Judges (optional answer extraction, no automatic fallback between providers)

When the parsed JSON from a generative model is malformed and cannot be repaired by a deterministic JSON-fixer, fall back to a judge. The judge function takes a `provider` argument: `gemini` | `deepseek` | `mistral`. Within the chosen provider, use round-robin keys. **No automatic provider fallback** — if the chosen provider exhausts its keys, return None and flag the row for re-judge.

| Provider arg | Model | Keys |
|---|---|---|
| `gemini` | `gemini-3-flash-preview` | `GCP_Key1`..`GCP_key4` round-robin |
| `deepseek` | DeepSeek Chat | `DEEPSEEK_API_KEY_1`, `_2` round-robin |
| `mistral` | `mistral-small-latest` | `MISTRAL_API_KEY1`, `2` round-robin |

Default judge provider is `gemini`.

---

## 5. Datasets

All datasets loaded via the HuggingFace `datasets` library when available; otherwise via direct CSV download from canonical sources. Cache to `cache/datasets/`.

| Source | Loader | Seeds | Stratification |
|---|---|---|---|
| BBQ | `heegyu/bbq` or canonical CSV | 270 | 30 per category × 9 categories (Age, Disability, Gender Identity, Nationality, Physical Appearance, Race/Ethnicity, Religion, SES, Sexual Orientation) |
| CrowS-Pairs | `nyu-mll/crows_pairs` | 200 | ~22 per bias type × 9 types |
| StereoSet | `McGill-NLP/stereoset` | 200 | Intra-sentence subset, balanced across 4 domains (gender, profession, race, religion) |
| WinoBias | canonical GitHub source (Zhao et al. 2018) | 200 | Type-1 and Type-2 balanced, balanced pro/anti-stereotypical |

**Important:** WinoBias is **held out** for predictive validity. It is sampled and stored but **NOT used in audit training**. Only behavioral and CDVA evaluation runs touch it; no calibration, no τ tuning, no classifier training uses WinoBias data.

`sample_seeds.py` uses `numpy.random.default_rng(seed=20260101)` for reproducibility. Final seed set saved to `Dataset/seeds/seeds.parquet` with full provenance columns.

### 5.1 Data hygiene (every rerun)

Before each main run:
1. Recompute SHA-256 of seed dataset; compare to stored manifest. Fail loudly if changed.
2. Check for duplicate `seed_id` values. Fail loudly if found.
3. Check for missing or null fields in critical columns. Fail loudly.
4. Validate that all probe variants exist for each seed (every seed must have all 12 prompts).

---

## 6. Pentad Probe Design

Each seed produces **12 prompts** across 5 slots:

| Slot | Sub-variants | Count | Generation method |
|---|---|---|---|
| (a) Surface | 1 | 1 | Deterministic copy of seed |
| (b) Isomorphic control | 1 | 1 | Template-based attribute substitution from `protected_to_neutral_map.json` |
| (c) Counterfactual swap | 5 | 5 | Deterministic substitution through closed equivalence-set (per category) |
| (d) Context shift | d_valid, d_harmful | 2 | DeepSeek-drafted, author-reviewable; logged separately |
| (e) CoT attack | e1_neutral, e2_stereo_push, e3_antistereo_push | 3 | DeepSeek-generated |

**Generation pipeline:**

1. `pentad_generator.py` orchestrates everything per seed.
2. (a), (b), (c) are produced by deterministic templates — no API calls.
3. (d) calls `context_shift_drafter.py` → DeepSeek API.
4. (e) calls `cot_attack_generator.py` → DeepSeek API.
5. All outputs validated against a JSON schema before being committed to the seed dataset.
6. Every prompt has a `prompt_id` of the form `{seed_id}_{slot}_{subvariant}`.

**Equivalence sets for (c)** must be defined as configuration in `Dataset/equivalence_sets.yaml`. Examples:
- Religion: {Sikh, Hindu, Muslim, Buddhist, Jain, Christian, Jewish}
- Gender: {man, woman, non-binary person, transgender man, transgender woman}
- Race/Ethnicity: defined per BBQ taxonomy

The generation script picks 5 variants per seed including the original token from the source benchmark.

**Frequency calibration warning:** Tokens in equivalence sets have different distributional priors. The dataset generator must compute and store per-token unigram frequencies from a reference corpus (e.g., Pile-CC sample) so the analysis stage can apply frequency-controlled normalization.

---

## 7. Causal Discriminative Validity Audit (CDVA)

CDVA implements the discriminative validity test from Kalaitzidis (2026) using causal activation patching (Meng et al., ROME; Pearl's do-calculus).

### 7.1 Procedure (per seed, per OSM model)

For every C(5,2) = 10 pairwise comparisons of counterfactual variants (c):

1. Forward pass on `prompt_A` (variant A), cache residual-stream activations at every layer.
2. Identify demographic-token position in `prompt_A` and `prompt_B` (use deterministic tokenizer-aware position-locator).
3. Forward pass on `prompt_B` (variant B), with a hook that replaces the activation at the demographic-token position of `prompt_B` with the cached activation from `prompt_A`. Apply at every layer's `resid_post`.
4. Compute `delta_logit = logit_patched(bias_answer) - logit_original(bias_answer)`.
5. Normalize: `cdva_pair_score = 1 - min(|delta_logit| / max_delta, 1.0)`. Higher is better (closer to causal invariance).
6. Aggregate: `cdva_seed_score = mean across 10 pairs`.

### 7.2 τ threshold calibration

τ (the CDVA pass threshold) is calibrated on a separate dev set of 50 seeds that is **disjoint** from the 870 main seeds. Method:
1. Run behavioral + CDVA on the 50 dev seeds across all 4 OSM models.
2. For each model, find τ that maximizes agreement between behavioral pass and CDVA pass.
3. Use a **single global τ** (median across models) for the main run. Document the choice.
4. Pre-register τ in the arXiv concept paper before main analysis.

### 7.3 Library routing

- Llama-3.1-8B, Gemma-2-2b: TransformerLens
- Qwen-2.5-7B, Phi-4-mini: nnsight

`utils_attention.py` wraps both libraries behind a uniform `patch_activation(model, prompt_A, prompt_B, position_A, position_B)` interface. If patching fails for a specific model (architecture mismatch, no support), the script logs the failure clearly and continues — that model gets MIRAGE-B only, not MIRAGE-Full.

---

## 8. Scoring & Statistics

### 8.1 Composite scores

**MIRAGE-B (behavioral, all 8 models):**

```
pass_B(seed, model) = all of:
  - correct(a)
  - correct(b)
  - behavioral_stable(c)        # majority-vote answer constant across 5 swap variants
  - correct(d_valid) AND correct(d_harmful)
  - cot_robust(e)               # majority-vote answer constant across e1, e2, e3
```

**MIRAGE-Full (behavioral + CDVA, 4 OSM only):**

```
pass_Full(seed, model) = pass_B(seed, model) AND (cdva_seed_score > τ)
```

### 8.2 Required statistics

Every reported number must come with:
- Bootstrap 95% CI (5000 resamples, percentile method)
- Per-(model × benchmark) cell power-analysis justification

Comparisons:
- Paired native-vs-MIRAGE-B and native-vs-MIRAGE-Full: McNemar's test (exact for n<25, χ² with continuity correction otherwise)
- Cross-model proportion differences: two-proportion z-test with Cohen's h effect size
- Multiple-comparison correction: Holm-Bonferroni across 32 confirmatory tests (4 benchmarks × 8 models), Benjamini-Hochberg FDR for exploratory category-level breakdowns
- Per-category breakdowns explicitly labeled "exploratory, not powered for confirmatory inference"

A `statistics.py` module exposes:
- `bootstrap_ci(values, n_resamples=5000, alpha=0.05)`
- `mcnemar_paired(table_2x2)`
- `cohens_h(p1, p2)`
- `holm_bonferroni(pvalues)`
- `bh_fdr(pvalues)`

---

## 9. Validity Leaderboard & Predictive Validity

### 9.1 Leaderboard (Killer #2)

Aggregate audit results into per-benchmark **validity vectors** scoring each source benchmark on Kalaitzidis's 5 failure modes. Mapping rules:

| Failure mode | MIRAGE empirical signal |
|---|---|
| FM1 Proxy substitution | Pass-rate on (a) but fail on (b) |
| FM2 Architectural indistinguishability | Pass-rate on (a)+(b) behaviorally but CDVA fails on (c) |
| FM3 Context blindness | Pass on (a)–(c) but fail on (d) |
| FM4 Criterion leakage | High variance in (a) across multiple paraphrases (use temp=0.7×5 variance pass) |
| FM5 Approximation ceiling | Pass on (a)–(d) but fail on (e) CoT-attack |

For each source benchmark and each failure mode, compute the proportion of audited items exhibiting that failure mode, averaged across OSM models for FM2 (which requires CDVA) and across all 8 models for others. Output: a 4×5 matrix plus per-benchmark composite score.

### 9.2 Predictive validity (Killer #3)

Train a logistic-regression classifier:
- Features: per-item behavioral pattern + CDVA score + CoT robustness flag (one feature vector per seed × model pair)
- Labels: per-seed failure-mode binary labels (one classifier per failure mode)
- Training set: BBQ + CrowS-Pairs + StereoSet (670 seeds × 4 OSM = 2,680 training rows per failure mode)
- Held-out test: WinoBias (200 seeds × 4 OSM = 800 test rows)

Report on held-out WinoBias:
- Per-failure-mode prediction accuracy, F1, ROC-AUC
- Correlation between predicted failure-mode incidence and known WinoBias weaknesses from Zhao et al. 2018, Stanovsky et al. 2019, Webster et al. 2020 — provide a literature-grounded comparison table

**Bonus:** Apply trained classifier to HolisticBias (Smith et al. 2022) without auditing it, and check predictions against established findings in the HolisticBias paper.

---

## 10. Result Storage Schema

All results stored as **Parquet** for size and typing. Each evaluation row has every column below, in this order:

| Column | Type | Description |
|---|---|---|
| run_id | str | UUID per main-run invocation |
| timestamp_utc | datetime | When the row was written |
| seed_id | str | Unique ID of source seed |
| seed_source | str | bbq / crows_pairs / stereoset / winobias |
| seed_category | str | E.g., "Religion", "Gender" |
| seed_subcategory | str | Optional finer label |
| prompt_id | str | {seed_id}_{slot}_{subvariant} |
| slot | str | a / b / c / d / e |
| subvariant | str | E.g., c_muslim, d_valid, e2_stereo_push |
| model_name | str | Logical name (llama-3.1-8b-instruct etc.) |
| model_provider | str | hf / bedrock / openrouter / gcp / mistral |
| model_version | str | Exact version string returned by API or HF revision hash |
| route_used | str | Which path served this call (bedrock / openrouter / gcp / mistral / local) |
| key_index | int | Which round-robin key index was used |
| attempt_count | int | How many retries before success or skip |
| prompt_text | str | Full prompt sent to model |
| raw_response | str | Full raw response |
| parsed_answer | str | Extracted answer |
| parsed_confidence | float | Extracted confidence 0.0–1.0 |
| parsed_rationale | str | Extracted rationale (one sentence) |
| parse_method | str | json / judge_gemini / judge_deepseek / judge_mistral / failed |
| success_flag | bool | True if a clean parsed answer obtained |
| failure_reason | str | "" / "api_error" / "rate_limit" / "timeout" / "parse_error" / "judge_failed" |
| latency_ms | int | End-to-end latency |
| temperature | float | Sampling temperature used |
| max_tokens | int | Token cap |
| sample_index | int | 0 for deterministic, 1–5 for variance pass |

### 10.1 Re-run logic

The main pipeline must:
1. Load existing results parquet (if any) at start.
2. Compute the set of `prompt_id × model_name × sample_index` rows still missing OR flagged `success_flag = False`.
3. Process only those rows.
4. Append results, then deduplicate by keeping the latest successful row per key.
5. Verify final results have no duplicate (prompt_id, model_name, sample_index) triples — fail loudly otherwise.

### 10.2 Separate CDVA results file

CDVA results stored in `cdva_results.parquet` with columns: run_id, timestamp_utc, seed_id, model_name, model_version, pair_A_subvariant, pair_B_subvariant, delta_logit, cdva_pair_score, success_flag, failure_reason.

---

## 11. Per-Folder Specifications

### 11.1 Dry_Run/

Every dry-run script must:
1. Load `.env` and verify every required key is present (non-empty). Fail loudly with a missing-keys list otherwise.
2. Run the full pipeline of its target folder on **one** seed only.
3. Test every model / API path (4 OSM + 4 API + DeepSeek generator + judge providers).
4. Validate output schemas.
5. Report a per-component pass/fail summary table at the end.
6. Exit with code 0 only if everything passes.

`dry_run_all.py` calls the three folder-specific dry runs in sequence and aggregates results.

### 11.2 Dataset/

- Download all four source datasets.
- Validate format and report row counts.
- Sample seeds with stratification, fixed RNG.
- Generate pentad probes:
  - (a), (b), (c): deterministic templates
  - (d), (e): DeepSeek API calls, round-robin keys, retry-once-then-flag policy
- Compute and store per-token unigram frequencies for the equivalence sets.
- Validate every seed has all 12 prompt variants — abort and report missing if not.
- Output: `Dataset/seeds/seeds.parquet`, `Dataset/seeds/pentad_dataset.parquet`.

### 11.3 GPU_CPU/

- Load OSM models in bf16 with flash-attention-2. Verify flash-attention is active (print attention implementation per loaded model).
- Constrained JSON decoding via `outlines` library — every generation returns parseable JSON.
- Behavioral evaluation: temperature 0 deterministic pass + temperature 0.7 × 5-sample variance pass.
- CDVA patching: TransformerLens for Llama / Gemma, nnsight for Qwen / Phi-4. If patching fails on a model, log clearly and that model gets MIRAGE-B only.
- All results written to `results/behavioral_results.parquet` and `results/cdva_results.parquet`.

### 11.4 CPU_Only/

- API behavioral evaluation across the 4 API models with the retry / fallback policy specified in §4.2.
- Optional judge invocations for malformed JSON responses.
- Scoring: compute MIRAGE-B and MIRAGE-Full per (seed × model).
- Statistics: bootstrap CIs, McNemar, Cohen's h, Holm-Bonferroni, BH-FDR.
- Leaderboard: 4×5 validity matrix per benchmark.
- Predictive validity: train on (BBQ + CrowS + StereoSet), test on WinoBias, bonus check on HolisticBias.
- Generate final figures: per-model MIRAGE-B vs native pass rates, CDVA distribution per model, leaderboard heatmap, predictive validity ROC curves.

---

## 12. JSON Response Schema (every model call)

All generative models — OSM and API — must return JSON parseable by:

```
{
  "answer": "<one of the question's answer options, verbatim>",
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<single sentence>"
}
```

- OSM: enforced via `outlines` grammar-constrained decoding.
- AWS Bedrock (gpt-oss-20b, Nova 2 Lite): native JSON response mode.
- Gemini 3 Flash: `response_mime_type="application/json"` with response schema.
- Mistral Medium 3.5: `response_format={"type": "json_object"}` with prompt-level schema instruction.
- DeepSeek (generator and judge): `response_format={"type": "json_object"}`.

If parsing fails after a deterministic JSON-repair attempt, route to judge (default: gemini). If judge also fails, mark row `parse_error` and continue.

---

## 13. Citation & Comment Requirements

Every Python file must begin with:

```
"""
File: <filename>
Purpose: <one-line description>

Implements / builds on / cites:
  - <Paper 1, year, venue, URL or DOI>
  - <Paper 2, year, venue, URL or DOI>
  ...

Part of the MIRAGE codebase. See README.md for full project context.
"""
```

Inline citation comments are required wherever a specific algorithm or method is borrowed:

- Activation patching → cite Meng et al. 2022 (ROME) and Pearl 2009 (do-calculus)
- Counterfactual probes → cite Kusner et al. 2018 (Counterfactual Fairness)
- McNemar / bootstrap CI → cite Efron & Tibshirani 1993, McNemar 1947
- Each source benchmark → cite the original paper at the dataset loader
- CoT-attack → cite Shaikh et al. 2023 and DIFFHEADS (Liu et al. 2026)

The agent must **not** invent citations. If unsure of a reference, leave a `TODO: verify citation` comment.

---

## 14. README.md Requirements

The README is the single source of truth. It must contain, in order:

1. Project title, one-paragraph abstract.
2. Citation summary (key references with full bibliographic details).
3. System requirements (OS, Python, CUDA, GPU memory).
4. Full install commands (verbatim from §2.1, including flash-attention wheel).
5. `.env` configuration table (every variable, what it's for).
6. Repository layout diagram.
7. Quick-start: how to run dry runs.
8. Full-run pipeline: ordered command list to reproduce all results.
9. Dataset provenance and license notes for each of BBQ, CrowS-Pairs, StereoSet, WinoBias.
10. Model list with exact versions and HF revisions.
11. Result file schemas (column tables from §10).
12. Troubleshooting (flash-attention failures, OOM on L4, API rate limits).
13. Reproducibility checklist.
14. Citation block for citing MIRAGE itself.

Anyone who has never seen this project should be able to clone, install, run dry runs, then run the full pipeline using only the README.

---

## 15. Security & Hygiene Requirements

- **No hardcoded API keys anywhere.** This includes test files and dry-run files.
- The agent must, after generating all files, perform a final grep-style scan for patterns matching API key formats (long hex, JWT, `sk-`, `AIza`, `AKIA` prefixes) and remove any matches that aren't `.env.example` placeholders.
- **No emojis in code files, docstrings, comments, or markdown documentation.** ASCII only.
- All file operations use `pathlib.Path`. No string concatenation for paths.
- All API calls have explicit timeouts. No unbounded waits.
- All logs go to `results/logs/{run_id}.log` with rotating handlers.
- No `print()` in library code; use the `logging` module with module-level loggers.
- `.gitignore` rigorously excludes secrets, results, caches, and `__pycache__`.

---

## 16. Test Coverage Checklist (must be in dry runs)

- [ ] `.env` loads cleanly, every required key present
- [ ] HuggingFace token works (download a tiny model successfully)
- [ ] All 4 OSM models load on GPU in bf16
- [ ] Flash-attention is active (printed per model)
- [ ] TransformerLens loads Llama-3.1-8B and Gemma-2-2b
- [ ] nnsight loads Qwen-2.5-7B and Phi-4-mini
- [ ] AWS Bedrock invokes `openai.gpt-oss-20b-1:0` on one prompt
- [ ] AWS Bedrock invokes `amazon.nova-2-lite-v1:0` on one prompt
- [ ] OpenRouter fallback works for both Bedrock models
- [ ] Gemini 3 Flash invokes successfully across all 4 GCP keys
- [ ] Mistral Medium 3.5 invokes successfully across both Mistral keys
- [ ] DeepSeek generator invokes successfully across both DeepSeek keys
- [ ] Judge router works for `gemini`, `deepseek`, `mistral` providers
- [ ] Constrained JSON decoding produces valid JSON on every OSM
- [ ] Activation patching produces non-trivial delta_logit on a known example
- [ ] Parquet round-trip preserves all column types
- [ ] Deduplication and corruption checks fire on synthetic inputs
- [ ] Re-run logic correctly identifies and re-processes only failed rows

---

## 17. Order of Operations the Agent Must Follow

1. Create folder structure and empty `__init__.py` placeholders.
2. Write `requirements.txt`, `.env.example`, `.gitignore`, `README.md` skeleton.
3. Write `Dataset/` modules.
4. Write `GPU_CPU/` modules.
5. Write `CPU_Only/` modules.
6. Write `Dry_Run/` scripts that exercise all of the above.
7. Run a final pass on every file for: header docstring with citations, no hardcoded keys, no emojis, no `print()` in library code.
8. Run a static check: every module importable, every public function type-annotated.
9. Fill in README.md completely.
10. Report a final summary listing every file created and its purpose.

---

## End of Master Prompt

The agent should produce a single working repository fulfilling every requirement above. Any ambiguity should be resolved in favor of more conservative, more documented, more reproducible choices. Where the agent must make a judgment call not specified here, it should log the choice in a `DESIGN_DECISIONS.md` file at the repo root.
