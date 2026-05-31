# MIRAGE — Mechanism-Indexed Reliability Audit for Group-bias Evaluation

MIRAGE is a discriminative-validity audit framework for LLM bias benchmarks. It operationalises the Epistematics methodology of Kalaitzidis (2026) by combining behavioural probing across five probe slots with causal activation patching (CDVA). Eight models — four open-source (OSM) and four API-served — are evaluated on 870 seeds drawn from BBQ, CrowS-Pairs, and StereoSet. WinoBias is held out for predictive-validity testing. Results are structured around Kalaitzidis's five failure modes (FM1–FM5) and reported with pre-registered statistical methods.

---

## Citation Summary

| Key reference | Details |
|---|---|
| Kalaitzidis (2026) | "The Evaluation Trap: Benchmark Design as Theoretical Commitment." arXiv:2605.14167 |
| Parrish et al. (2022) | "BBQ: A Hand-Built Bias Benchmark for Question Answering." Findings of ACL 2022. |
| Nangia et al. (2020) | "CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models." EMNLP 2020. |
| Nadeem et al. (2021) | "StereoSet: Measuring stereotypical bias in pretrained language models." ACL-IJCNLP 2021. |
| Zhao et al. (2018) | "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods." NAACL 2018. (WinoBias) |
| Shaikh et al. (2023) | "On Second Thought, Let's Not Think Step by Step! Bias and Toxicity in Zero-Shot Reasoning." ACL 2023. |
| Meng et al. (2022) | "Locating and Editing Factual Associations in GPT." NeurIPS 2022. (ROME, activation patching) |
| Pearl (2009) | *Causality.* Cambridge University Press. (do-calculus framing) |
| Efron & Tibshirani (1993) | *An Introduction to the Bootstrap.* Chapman & Hall. |
| Bean et al. (2025) | "Measuring what Matters: Construct Validity in Large Language Model Benchmarks." NeurIPS 2025 Datasets and Benchmarks Track. |
| Wang et al. (2025) | "Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs." ACL 2025. |

---

## System Requirements

| Requirement | Specification |
|---|---|
| Operating system | Ubuntu 22.04 or 24.04 LTS, x86_64 only |
| Python | 3.12 (cp312 wheel required for flash-attention) |
| CUDA | 12.4 |
| GPU | NVIDIA L4 24 GB or equivalent single-GPU |
| RAM | >= 32 GB recommended |
| Disk | >= 100 GB for model weights, datasets, and results |

**Windows and macOS are not supported.** Flash-attention-2 is Linux/x86_64-only. The pipeline will print a clear error and exit immediately on any other platform.

---

## Installation

Run all commands in order on a fresh Ubuntu 22.04/24.04 machine with CUDA 12.4 already installed.

```bash
python3 -m pip install --upgrade pip setuptools wheel \
  && python3 -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
  && python3 -m pip install numpy<2.0 transformers==4.46.0 accelerate==0.34.0 datasets==2.16.0 \
       bitsandbytes==0.46.1 pandas==2.2.2 tqdm==4.65.0 python-dotenv==1.0.0 requests==2.31.0 \
       sentencepiece==0.2.0 protobuf==4.25.0 \
  && wget -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl -O /tmp/flash_attn.whl \
  && python3 -m pip install --no-deps /tmp/flash_attn.whl
```

Install remaining dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Verify the installation:

```bash
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

## Environment Variables

Copy `.env.example` to `.env` at the repository root and fill in every value before running.

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `HUGGINGFACE_TOKEN` | HuggingFace model downloads (must have access to Llama-3.1-8B-Instruct and Gemma-2-2b-it) |
| `AWS_BEDROCK_KEY` | Base64-encoded AWS credentials for Bedrock (covers gpt-oss-20b-1:0 and nova-2-lite-v1:0) |
| `OPENROUTER_API_KEY_1` | OpenRouter fallback, round-robin slot 1 |
| `OPENROUTER_API_KEY_2` | OpenRouter fallback, round-robin slot 2 |
| `OPENROUTER_API_BASE_URL` | OpenRouter API base URL (default: https://openrouter.ai/api/v1) |
| `GEMINI_API_KEY_1` | GCP Gemini key 1 (four-key round-robin) |
| `GEMINI_API_KEY_2` | GCP Gemini key 2 |
| `GEMINI_API_KEY_3` | GCP Gemini key 3 |
| `GEMINI_API_KEY_4` | GCP Gemini key 4 |
| `GEMINI_MODEL_NAME` | Gemini model name (e.g., gemini-2.5-flash-lite) |
| `MISTRAL_API_KEY1` | Mistral platform key 1 (two-key round-robin) |
| `MISTRAL_API_KEY2` | Mistral platform key 2 |
| `MISTRAL_MODEL_NAME` | Mistral model name (e.g., mistral-medium-latest) |
| `DEEPSEEK_API_KEY_1` | DeepSeek key 1 — used for pentad template generation only, never evaluation |
| `DEEPSEEK_API_KEY_2` | DeepSeek key 2 |
| `DEEPSEEK_API_BASE_URL` | DeepSeek API base URL |
| `DEEPSEEK_PRIMARY_MODEL_NAME` | DeepSeek model for generation (e.g., deepseek-chat) |
| `DEEPSEEK_JUDGE_MODEL_NAME` | DeepSeek model for judging (may differ from primary) |

**Security:** `.env` is git-ignored. Never commit real keys. Use `.env.example` as the template.

---

## Repository Layout

```
mirage/
├── README.md                    # This file
├── requirements.txt             # Pinned dependencies
├── .env.example                 # Key template (no real values)
├── .gitignore
├── config.py                    # Central configuration loader
├── logger_setup.py              # Rotating file logger
├── DESIGN_DECISIONS.md          # Judgment calls and rationale
│
├── Dry_Run/
│   ├── dry_run_dataset.py       # Dataset pipeline check
│   ├── dry_run_gpu_cpu.py       # OSM model + CDVA check
│   ├── dry_run_cpu_only.py      # API + scoring check
│   └── dry_run_all.py           # Master dry run (runs all three)
│
├── Dataset/
│   ├── equivalence_sets.yaml    # Closed equivalence sets for counterfactual slot (c)
│   ├── download_bbq.py
│   ├── download_crows_pairs.py
│   ├── download_stereoset.py
│   ├── download_winobias.py
│   ├── sample_seeds.py          # Stratified seed selection (RNG seed=20260101)
│   ├── pentad_generator.py      # Orchestrates all 12 probe variants per seed
│   ├── cot_attack_generator.py  # Slot (e) via DeepSeek API
│   ├── context_shift_drafter.py # Slot (d) via DeepSeek API
│   └── validate_pentad.py       # Schema + completeness validation
│
├── GPU_CPU/
│   ├── load_osm.py              # bf16 + flash-attention-2 loader
│   ├── osm_behavioral.py        # Behavioural evaluation, 4 OSM models
│   ├── cdva_patching.py         # Causal activation patching, 10 pairs/seed
│   ├── cdva_calibration.py      # tau threshold calibration on 50-seed dev set
│   └── utils_attention.py       # Unified TransformerLens / nnsight interface
│
├── CPU_Only/
│   ├── api_clients/
│   │   ├── bedrock_client.py    # AWS Bedrock + OpenRouter fallback
│   │   ├── gemini_client.py     # GCP Gemini 4-key round-robin
│   │   ├── mistral_client.py    # Mistral 2-key round-robin
│   │   └── openrouter_client.py # OpenRouter 2-key round-robin
│   ├── api_behavioral.py        # API model evaluation
│   ├── judge_router.py          # Malformed-JSON recovery via judge model
│   ├── scoring.py               # MIRAGE-B, MIRAGE-Full composite scores
│   ├── statistics.py            # Bootstrap CI, McNemar, Cohen's h, corrections
│   ├── leaderboard.py           # 4x5 FM validity matrix
│   ├── predictive_validity.py   # Logistic classifier, held-out WinoBias test
│   └── results_analysis.py      # Final figures and tables
│
└── results/                     # All output (gitignored)
    ├── pentad_dataset.parquet
    ├── behavioral_results.parquet
    ├── cdva_results.parquet
    ├── scored_results.parquet
    ├── leaderboard.parquet
    ├── tau_calibration.json
    └── figures/
```

---

## Quick-Start: Dry Runs

Run this before attempting the full pipeline. Dry runs validate every component on one seed.

```bash
# All three phases (dataset + GPU/OSM + API)
python3 Dry_Run/dry_run_all.py

# GPU-only environment (skip OSM model loading)
python3 Dry_Run/dry_run_all.py --skip-gpu

# Individual phase
python3 Dry_Run/dry_run_all.py --only dataset
python3 Dry_Run/dry_run_all.py --only gpu_cpu
python3 Dry_Run/dry_run_all.py --only cpu_only
```

Exit code 0 means all checks passed. Any non-zero exit means at least one component failed; review the log in `results/logs/`.

---

## Full-Run Pipeline

Run these commands in order. Each step is resume-capable — re-running after a crash or API timeout will pick up from where it stopped.

```bash
# Step 1 — Download all four source datasets and validate
python3 -c "
from Dataset.download_bbq import download_bbq, validate_bbq
from Dataset.download_crows_pairs import download_crows_pairs, validate_crows_pairs
from Dataset.download_stereoset import download_stereoset, validate_stereoset
from Dataset.download_winobias import download_winobias, validate_winobias
validate_bbq(download_bbq())
validate_crows_pairs(download_crows_pairs())
validate_stereoset(download_stereoset())
validate_winobias(download_winobias())
print('All datasets validated.')
"

# Step 2 — Sample 870 main seeds + 50 dev seeds (deterministic, RNG=20260101)
python3 -c "
from Dataset.sample_seeds import sample_seeds, verify_seeds_integrity
main, dev = sample_seeds()
verify_seeds_integrity()
print(f'Seeds: {len(main)} main, {len(dev)} dev')
"

# Step 3 -- Generate pentad dataset (slots a/b/c deterministic; d/e via DeepSeek)
#            Resume-capable: if interrupted, restart and it picks up from checkpoint.
python3 run_dataset.py

# Step 3b (recovery only) -- If slot-c counts were wrong after a failed build,
#           patch deterministic slots without re-calling any API:
python3 patch_det_slots.py

# Step 4 -- Validate pentad completeness (12 prompts per seed)
python3 -c "
import pandas as pd
from Dataset.validate_pentad import run_all_validations
from config import RESULTS_DIR
df = pd.read_parquet(RESULTS_DIR / 'pentad_dataset.parquet')
run_all_validations(df)
print('Pentad dataset valid.')
"

# Step 5 -- OSM behavioural evaluation (GPU required)
python3 -c "
import pandas as pd
from GPU_CPU.load_osm import load_all_osm_models
from GPU_CPU.osm_behavioral import run_osm_behavioral
from config import RESULTS_DIR, OSM_MODELS
pentad_df = pd.read_parquet(RESULTS_DIR / 'pentad_dataset.parquet')
models = load_all_osm_models()
run_osm_behavioral(pentad_df, models, run_id='main_run')
"

# Step 6 -- CDVA causal activation patching (GPU required)
python3 -c "
import pandas as pd
from GPU_CPU.load_osm import load_all_osm_models
from GPU_CPU.cdva_patching import run_cdva
from config import RESULTS_DIR
pentad_df = pd.read_parquet(RESULTS_DIR / 'pentad_dataset.parquet')
models = load_all_osm_models()
run_cdva(pentad_df, models, run_id='main_run')
"

# Step 7 -- CDVA tau calibration on 50-seed dev set
python3 -c "
import pandas as pd
from GPU_CPU.cdva_calibration import calibrate_tau
from config import RESULTS_DIR
behavioral_dev = pd.read_parquet(RESULTS_DIR / 'behavioral_results_dev.parquet')
cdva_dev = pd.read_parquet(RESULTS_DIR / 'cdva_results_dev.parquet')
tau = calibrate_tau(behavioral_dev, cdva_dev)
print(f'Calibrated tau = {tau:.4f}')
"

# Step 8 -- API model behavioural evaluation (CPU, requires valid API keys)
python3 -c "
import pandas as pd
from CPU_Only.api_behavioral import run_api_behavioral
from config import RESULTS_DIR
pentad_df = pd.read_parquet(RESULTS_DIR / 'pentad_dataset.parquet')
run_api_behavioral(pentad_df, run_id='main_run')
"

# Step 9 -- Score all results (MIRAGE-B and MIRAGE-Full)
python3 -c "
import pandas as pd
from CPU_Only.scoring import score_all
from GPU_CPU.cdva_calibration import load_tau
from config import RESULTS_DIR
behavioral = pd.read_parquet(RESULTS_DIR / 'behavioral_results.parquet')
cdva = pd.read_parquet(RESULTS_DIR / 'cdva_results.parquet')
tau = load_tau()
score_all(behavioral, cdva, tau)
"

# Step 10 -- Build validity leaderboard
python3 -c "
import pandas as pd
from CPU_Only.leaderboard import build_leaderboard
from config import RESULTS_DIR
behavioral = pd.read_parquet(RESULTS_DIR / 'behavioral_results.parquet')
cdva = pd.read_parquet(RESULTS_DIR / 'cdva_results.parquet')
build_leaderboard(behavioral, cdva)
"

# Step 11 -- Predictive validity (WinoBias held-out test)
python3 -c "
import pandas as pd
from CPU_Only.predictive_validity import run_predictive_validity
from config import RESULTS_DIR
behavioral = pd.read_parquet(RESULTS_DIR / 'behavioral_results.parquet')
cdva = pd.read_parquet(RESULTS_DIR / 'cdva_results.parquet')
results = run_predictive_validity(behavioral, cdva)
for fm, metrics in results.items():
    print(f'{fm}: acc={metrics[\"accuracy\"]:.3f} f1={metrics[\"f1\"]:.3f} auc={metrics[\"roc_auc\"]:.3f}')
"

# Step 12 -- Generate all figures
python3 -c "
import pandas as pd
from CPU_Only.results_analysis import run_results_analysis
run_results_analysis()
"
```

---

## Dataset Provenance and Licences

| Dataset | HuggingFace ID | Licence | Original paper |
|---|---|---|---|
| BBQ | `heegyu/bbq` | CC BY 4.0 | Parrish et al. ACL Findings 2022 |
| CrowS-Pairs | `nyu-mll/crows_pairs` | CC BY SA 4.0 | Nangia et al. EMNLP 2020 |
| StereoSet | `McGill-NLP/stereoset` | MIT | Nadeem et al. ACL-IJCNLP 2021 |
| WinoBias | GitHub (Zhao et al. 2018) | MIT | Zhao et al. NAACL 2018 |

WinoBias is used exclusively as a held-out test for predictive validity. It is not used in calibration or classifier training.

---

## Model List

| Slot | Model ID | HuggingFace revision | Patching library |
|---|---|---|---|
| OSM-1 | `meta-llama/Llama-3.1-8B-Instruct` | Latest at download | TransformerLens |
| OSM-2 | `Qwen/Qwen2.5-7B-Instruct` | Latest at download | nnsight |
| OSM-3 | `google/gemma-2-2b-it` | Latest at download | TransformerLens |
| OSM-4 | `microsoft/Phi-4-mini-instruct` | Latest at download | nnsight |
| API-1 | `openai.gpt-oss-20b-1:0` | Bedrock / OpenRouter | N/A |
| API-2 | `amazon.nova-2-lite-v1:0` | Bedrock / OpenRouter | N/A |
| API-3 | `GEMINI_MODEL_NAME` (env) | GCP Gemini | N/A |
| API-4 | `MISTRAL_MODEL_NAME` (env) | Mistral platform | N/A |
| Generator | `DEEPSEEK_PRIMARY_MODEL_NAME` (env) | DeepSeek platform | NOT evaluated |

The generator (DeepSeek) is used only for slot (d) and (e) template generation. It does not appear in the evaluation pipeline.

---

## Result File Schemas

### behavioral_results.parquet

| Column | Type | Description |
|---|---|---|
| run_id | str | UUID per main-run invocation |
| timestamp_utc | datetime | Row write time (UTC) |
| seed_id | str | Unique seed identifier |
| seed_source | str | bbq / crows_pairs / stereoset / winobias |
| seed_category | str | E.g., Religion, Gender |
| seed_subcategory | str | Optional finer label |
| prompt_id | str | {seed_id}_{slot}_{subvariant} |
| slot | str | a / b / c / d / e |
| subvariant | str | E.g., c_muslim, d_valid, e2_stereo_push |
| model_name | str | Logical model name |
| model_provider | str | hf / bedrock / openrouter / gcp / mistral |
| model_version | str | Exact version string |
| route_used | str | bedrock / openrouter / gcp / mistral / local |
| key_index | int | Round-robin key index |
| attempt_count | int | Retries before success or skip |
| prompt_text | str | Full prompt sent |
| raw_response | str | Full raw response |
| parsed_answer | str | Extracted answer |
| parsed_confidence | float | Extracted confidence 0.0-1.0 |
| parsed_rationale | str | Extracted rationale |
| parse_method | str | json / judge_gemini / judge_deepseek / judge_mistral / failed |
| success_flag | bool | True if clean parsed answer obtained |
| failure_reason | str | "" / api_error / rate_limit / timeout / parse_error / judge_failed |
| latency_ms | int | End-to-end latency in ms |
| temperature | float | Sampling temperature |
| max_tokens | int | Token cap |
| sample_index | int | 0=deterministic, 1-5=variance pass |

### cdva_results.parquet

| Column | Type | Description |
|---|---|---|
| run_id | str | UUID per main-run invocation |
| timestamp_utc | datetime | Row write time (UTC) |
| seed_id | str | Unique seed identifier |
| model_name | str | Logical model name |
| model_version | str | Exact version string |
| pair_A_subvariant | str | Counterfactual variant A |
| pair_B_subvariant | str | Counterfactual variant B |
| delta_logit | float | Patched minus original logit |
| cdva_pair_score | float | 1 - min(|delta_logit|/max_delta, 1.0) |
| success_flag | bool | True if patching succeeded |
| failure_reason | str | "" or error description |

### scored_results.parquet

| Column | Type | Description |
|---|---|---|
| seed_id | str | Unique seed identifier |
| model_name | str | Logical model name |
| mirage_b_pass | bool | Passed MIRAGE-B |
| mirage_full_pass | bool | Passed MIRAGE-Full (OSM only) |
| cdva_seed_score | float | Mean CDVA score (OSM only) |

---

## Troubleshooting

### Flash-attention fails to import

Check that CUDA 12.4 is installed and that the wheel filename matches your exact PyTorch and CUDA versions:

```bash
python3 -c "import torch; print(torch.version.cuda)"
```

If CUDA version differs from 12.4, download the matching wheel from:
https://github.com/Dao-AILab/flash-attention/releases

### Out-of-memory on L4 24 GB

Each OSM model is loaded in bf16. Models are unloaded between CDVA patch runs. If OOM still occurs:

1. Verify no other process is using the GPU: `nvidia-smi`
2. Reduce the number of concurrent CDVA pairs by editing `cdva_patching.py`.
3. Check `accelerate` device_map configuration in `load_osm.py`.

### API rate limits

Each client uses round-robin key rotation and a retry-once policy. If rate limits persist:

1. Increase the per-call sleep in the relevant client.
2. Add additional API keys to `.env` (not currently supported without code changes).
3. Use the `--only` flag on `dry_run_all.py` to test individual API routes.

### Bedrock credential errors

`AWS_BEDROCK_KEY` must be base64-encoded credentials in the format expected by `bedrock_client.py`. Confirm decoding produces valid JSON with `access_key_id` and `secret_access_key` fields.

---

## Reproducibility Checklist

Before submitting results:

- [ ] `.env` loaded cleanly, all keys present (`python3 Dry_Run/dry_run_all.py`)
- [ ] Seed SHA-256 matches stored manifest (`Dataset/sample_seeds.py::verify_seeds_integrity()`)
- [ ] No duplicate seed_ids
- [ ] All 12 probe variants present for every seed
- [ ] Tau value pre-registered (stored in `results/tau_calibration.json`)
- [ ] `run_id` recorded for every result row
- [ ] OSM model HuggingFace revision hashes logged
- [ ] API model version strings captured per row
- [ ] RNG seed 20260101 used throughout (no calls to `random.seed()` or `np.random.seed()`)
- [ ] All statistical tests used Holm-Bonferroni for confirmatory and BH-FDR for exploratory comparisons
- [ ] Bootstrap CIs computed with 5000 resamples

---

## Citing MIRAGE

If you use MIRAGE in your research, please cite:

```bibtex
@article{mirage2026,
  title   = {{MIRAGE}: Mechanism-Indexed Reliability Audit for Group-bias Evaluation},
  author  = {Deb, Koushik and others},
  journal = {TODO: update with journal and year},
  year    = {2026},
  note    = {Preprint. TODO: update with DOI.}
}
```

Also cite the Epistematics framework this work operationalises:

```bibtex
@article{kalaitzidis2026,
  title   = {The Evaluation Trap: Benchmark Design as Theoretical Commitment},
  author  = {Kalaitzidis, Athanasios},
  journal = {arXiv preprint arXiv:2605.14167},
  year    = {2026}
}
```
