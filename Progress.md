# MIRAGE Project — Progress & Handoff Document

**Last updated:** June 5, 2026  
**Status:** GPU pipeline **complete**. API (CPU) pipeline **not started**. Paper draft **not written**.  
**Target venue:** IEEE Transactions on Computational Social Systems (TCSS)  
**GitHub:** https://github.com/DevDaring/Audit_Benchmark (branch `main`, commit includes GPU results + Analysis)

---

## 1. What This Project Is

MIRAGE (Mechanism-Indexed Reliability Audit for Group-bias Evaluation) is a discriminative-validity audit framework for LLM bias benchmarks. It evaluates models on a **pentad probe** (5 slots × 12 prompts per seed) drawn from BBQ, CrowS-Pairs, and StereoSet (596 seeds, 7,152 prompt rows).

**Eight models total:**
- **4 OSM (open-source, GPU):** Llama-3.1-8B, Qwen2.5-7B, Gemma-2-2B, Phi-4-mini — behavioral + CDVA (causal activation patching)
- **4 API (cloud, CPU only):** qwen3-next-80b-a3b (Bedrock→OpenRouter), amazon-nova-2-lite (Bedrock→OpenRouter), gemini-2.5-flash (LinkAPI→OpenRouter→MegaLLM), mistral-medium (Mistral→OpenRouter) — behavioral only (no CDVA)

**Core metrics:**
- **MIRAGE-B** — behavioral validity (slots a–e, gold-answer based)
- **MIRAGE-Full** — MIRAGE-B + CDVA pass (OSM models only)
- **Validity gap** — native benchmark pass rate minus MIRAGE-Full pass rate
- **FM1–FM5** — five failure modes (proxy substitution, architectural indistinguishability, context blindness, criterion leakage, CoT failure)

Full theory and methodology: `Code/mirage/README.md`

---

## 2. What Has Been Completed

### 2.1 Dataset & Pentad (pre-GPU)

| Item | Status |
|------|--------|
| Pentad dataset (596 seeds × 12 variants = 7,152 rows) | Done |
| Source benchmarks: BBQ (254), CrowS-Pairs (181), StereoSet (161) | Done |
| Slot-b grammar patch, slots d/e regeneration (DeepSeek) | Done |
| `assert_production_ready()` validation gate | Passed |
| WinoBias held out (predictive validity — not yet evaluated) | By design |

**File:** `Code/mirage/Dataset/seeds/pentad_dataset.parquet`

### 2.2 GPU Pipeline (GCP A100 40 GB) — COMPLETE

Ran on GCP VM `audit` (us-central1-f). **VM can be deleted** — all results downloaded locally and pushed to GitHub.

| Output | Rows | Status |
|--------|------|--------|
| `behavioral_results.parquet` (OSM only) | 40,528 (4 × 10,132) | Complete |
| `cdva_results.parquet` | 23,840 (4 × 5,960) | Complete, 100% success |
| Runtime | 16.7 hours total | Jun 5, 2026 |

**Data quality (verified):**
- CDVA `position_fallback_used` = 0.00%
- Zero `delta_logit` rate = 6.20% (genuine non-causal, not a bug)
- No duplicates, no NaN/Inf
- 596/596 seeds per model

**Critical bugs fixed during GPU run** (document in paper methods/limitations):
1. nnsight proxy chain for Qwen/Phi CDVA (`AttributeError: no attribute 'output'`)
2. CDVA position detection for multi-word swap tokens (53% fallback → 0%)
3. Phi/Qwen behavioral `batch_size=1` slowdown (32h → 3h)
4. `device_map="auto"` CPU offload (37× slowdown)
5. Gemma-2 TransformerLens device mismatch

Details: `Help/VM_progress.md`, `Code/mirage/README.md` §15

### 2.3 Results Downloaded & Pushed to GitHub

| File | Local path | On GitHub |
|------|------------|-----------|
| `behavioral_results.parquet` | `Code/mirage/results/` | Yes (force-added) |
| `cdva_results.parquet` | `Code/mirage/results/` | Yes |
| `pentad_dataset.parquet` | `Code/mirage/Dataset/seeds/` | Yes |

### 2.4 CPU Post-Processing (OSM results only)

Ran locally on laptop Python (no API calls):

| Output | Location |
|--------|----------|
| `scored_results.parquet` | `Code/mirage/results/` |
| `leaderboard.parquet` | `Code/mirage/results/` |
| `tau_calibration.json` (τ = 0.7644, 75th percentile) | `Code/mirage/results/` |
| `validity_gap_leaderboard.md` | `Code/mirage/results/` |
| Figures (3 PNGs) | `Code/mirage/results/figures/` |

**Note on tau:** Dev seeds are not tagged in the dataset. Tau uses the 75th percentile of |delta_logit| across all CDVA rows. Disclose this in the paper methods section. Proper dev-set calibration requires tagging dev seeds and running `GPU_CPU/cdva_calibration.py`.

### 2.5 Analysis Folder (publication-oriented)

Full analysis pipeline in `Analysis/` — independent of `CPU_Only/`, produces richer tables and 6 figures.

| Script | Output |
|--------|--------|
| `01_scoring.py` | MIRAGE-B / MIRAGE-Full per seed × model |
| `02_failure_modes.py` | FM1–FM5 rates, leaderboard matrix |
| `03_cdva_commutators.py` | Algebraic CDVA commutator stats |
| `04_algebraic_validity.py` | PAV axioms A1–A6, measurement laws M1–M5 |
| `05_statistical_tests.py` | Bootstrap CIs, McNemar, Cohen's h |
| `06_figures.py` | 6 publication figures (300 DPI) |
| `run_all.py` | Master runner |

**Report:** `Analysis/analysis.md` (headline findings, tables, algebraic interpretation)  
**Outputs:** `Analysis/outputs/` (parquet, JSON, figures)

Run: `python Analysis/run_all.py` from repo root (uses system Python + pandas/numpy/scipy/matplotlib/seaborn).

### 2.6 API (CPU) Pipeline — NOT STARTED

0 API model rows exist. The 4 API models still need ~40,528 sequential API calls (~15–30+ hours depending on latency). Code is ready; run was stopped because laptop could not stay awake.

---

## 3. Repository Folder Guide

```
Audit_Benchmark/
│
├── Progress.md                    ← THIS FILE (handoff document)
│
├── Code/
│   ├── MIRAGE_MASTER_PROMPT.md    # High-level project prompt
│   └── mirage/                    # ★ MAIN CODEBASE
│       ├── README.md              # ★ Full research doc — read first
│       ├── .env                   # API keys (NOT on GitHub — create locally)
│       ├── config.py              # Models, paths, keys
│       │
│       ├── Dataset/               # Pentad construction
│       │   ├── seeds/
│       │   │   └── pentad_dataset.parquet   # 7,152 rows
│       │   ├── pentad_generator.py
│       │   ├── validate_pentad.py
│       │   └── download_*.py      # BBQ, CrowS, StereoSet sources
│       │
│       ├── GPU_CPU/               # GPU pipeline (DONE — do not re-run unless fixing data)
│       │   ├── run_gpu_pipeline.py
│       │   ├── osm_behavioral.py
│       │   ├── cdva_patching.py
│       │   ├── utils_attention.py # CDVA patching (TransformerLens + nnsight)
│       │   └── load_osm.py
│       │
│       ├── CPU_Only/              # ★ CPU pipeline (API + scoring)
│       │   ├── api_behavioral.py  # 4 API models, sequential, checkpoint every 50
│       │   ├── scoring.py         # MIRAGE-B, MIRAGE-Full
│       │   ├── leaderboard.py     # FM1–FM5 matrix
│       │   ├── validity_gap_table.py
│       │   ├── statistics.py
│       │   ├── predictive_validity.py
│       │   ├── results_analysis.py
│       │   └── api_clients/       # Bedrock, Gemini, Mistral, OpenRouter
│       │
│       ├── Dry_Run/               # Sanity checks before production
│       │   ├── dry_run_all.py
│       │   └── dry_run_cpu_only.py
│       │
│       ├── results/               # ★ PRODUCTION OUTPUTS
│       │   ├── behavioral_results.parquet   # 40,528 OSM rows (+ API when run)
│       │   ├── cdva_results.parquet         # 23,840 rows
│       │   ├── scored_results.parquet
│       │   ├── leaderboard.parquet
│       │   ├── tau_calibration.json
│       │   ├── validity_gap_leaderboard.md
│       │   └── figures/
│       │
│       ├── run_cpu_full.py        # ★ API behavioral + post-processing
│       └── run_cpu_postprocess.py # Scoring only (no API calls)
│
├── Analysis/                      # ★ Publication analysis (OSM results done)
│   ├── analysis.md                # Full findings report
│   ├── run_all.py
│   ├── 01_scoring.py … 06_figures.py
│   └── outputs/                   # Tables, figures, JSON
│
├── Submission/                    # ★ Paper writing
│   ├── MIRAGE_PAPER_PROMPT_TCSS.md  # ★ Detailed paper instructions + data facts
│   ├── PAPER_REVIEW_INSTRUCTIONS.md
│   ├── IEEE_TCSS.tex              # LaTeX draft (started)
│   └── references.bib
│
├── Help/                          # Ops documentation
│   ├── VM_progress.md             # GPU run log, bugs, resume rules
│   ├── GCP_GPU_Setup.md           # GCP VM setup (historical)
│   └── Akash_Deployment.md        # Akash GPU deployment (historical)
│
└── akash/                         # VM deployment scripts (historical, not needed for CPU run)
```

---

## 4. Key Results (OSM only — for paper tables)

From `Analysis/analysis.md` and `Code/mirage/results/validity_gap_leaderboard.md`:

| Metric | Value |
|--------|-------|
| MIRAGE-B pass rate (all OSM) | 13.3% |
| MIRAGE-Full pass rate (OSM only) | 11.5% |
| Tau (75th pct \|delta_logit\|) | 0.7644 |

**Validity gap (native − MIRAGE-Full), macro-averaged:**
- BBQ: **39.4%**
- CrowS-Pairs: **20.4%**
- StereoSet: **17.2%**

**Per model (MIRAGE-B / MIRAGE-Full):**
- Llama-3.1-8B: 25.0% / 24.3%
- Qwen2.5-7B: 17.6% / 11.9%
- Gemma-2-2B: 10.6% / 9.7%
- Phi-4-mini: 0.0% / 0.0%

**CDVA:** Race and gender show highest commutator magnitudes. FM4 (criterion leakage) is the dominant failure mode across benchmarks.

**Mandatory CDVA filter for all analysis:** `position_fallback_used == False` (all 23,840 production rows satisfy this).

---

## 5. What the Colleague Must Do Next

### Phase A — Set Up Personal VM for CPU/API Run

**Requirements:** Linux or Windows with Python 3.10–3.12. No GPU needed. Stable network for ~40k API calls.

#### Step 1: Clone repo and install dependencies

```bash
git clone https://github.com/DevDaring/Audit_Benchmark.git
cd Audit_Benchmark/Code/mirage
```

Create `.env` in `Code/mirage/` (copy from team — **not on GitHub**). Required keys:

```
HUGGINGFACE_TOKEN=...
DEEPSEEK_API_KEY_1=...
DEEPSEEK_API_KEY_2=...
OPENROUTER_API_KEY_1=...
OPENROUTER_API_KEY_2=...
GEMINI_API_KEY_1=... through GEMINI_API_KEY_4=...
AWS_ACCESS_KEY=...
AWS_SECRET_KEY=...
MISTRAL_API_KEY1=...
MISTRAL_API_KEY2=...
```

Install Python packages:

```bash
pip install pandas pyarrow numpy scipy matplotlib seaborn \
  python-dotenv openai google-generativeai mistralai boto3 httpx scikit-learn json-repair
```

#### Step 2: Verify GPU results are present

```bash
cd Code/mirage
python -c "
import pandas as pd
b = pd.read_parquet('results/behavioral_results.parquet')
c = pd.read_parquet('results/cdva_results.parquet')
print('behavioral', len(b), 'models', sorted(b.model_name.unique()))
print('cdva', len(c), 'fallback', c.position_fallback_used.sum())
assert len(b) == 40528
assert len(c) == 23840
print('OK')
"
```

Expected: 40,528 behavioral (4 OSM models), 23,840 CDVA, 0 position fallbacks.

#### Step 3: Dry run (mandatory before API production)

```bash
python Dry_Run/dry_run_cpu_only.py
```

All checks must PASS (Bedrock qwen3-next + nova, DeepSeek, Mistral, OpenRouter fallback, judges, statistics). Round-robin keys absorb per-key rate limits — still OK if the dry run passes overall.

#### Step 4: Run full CPU pipeline (API + post-processing)

```bash
# Run in tmux/screen/nohup — takes many hours
nohup python run_cpu_full.py > results/logs/cpu_full_run.log 2>&1 &
```

**What this does:**
1. **Phase 1:** API behavioral for 4 models (appends to `behavioral_results.parquet`)
   - Sequential, one API call at a time (no parallelism — avoids rate limits)
   - Checkpoints every 50 prompts — safe to stop and resume
   - ~40,528 API calls total (~10,132 per model)
2. **Phase 2:** Re-score all 8 models, leaderboard, validity gap, figures

**Monitor progress:**

```bash
tail -f results/logs/cpu_full_run.log
python -c "import pandas as pd; b=pd.read_parquet('results/behavioral_results.parquet'); print(len(b), 'rows'); print(b.groupby('model_name').size())"
```

**Resume after interrupt:** Run the same command again. Completed rows (with `success_flag=True`) are skipped automatically.

**Expected final behavioral row count:** 40,528 (OSM) + 40,528 (API) = **81,056 rows**

#### Step 5: Re-run analysis after API completes

```bash
cd ../..   # repo root
python Analysis/run_all.py
```

Update `Analysis/analysis.md` with API model results. Re-run `python run_cpu_postprocess.py` if using `CPU_Only/` outputs for paper tables in `results/`.

---

### Phase B — Paper Writing

#### Primary resources (read in this order)

1. **`Code/mirage/README.md`** — full methodology, PAV framework, CDVA, failure modes
2. **`Submission/MIRAGE_PAPER_PROMPT_TCSS.md`** — paper structure, known data facts, section-by-section instructions, mandatory filters
3. **`Analysis/analysis.md`** — computed findings, tables, figure references
4. **`Submission/PAPER_REVIEW_INSTRUCTIONS.md`** — style and review criteria
5. **`Submission/IEEE_TCSS.tex`** — started LaTeX draft
6. **`Submission/references.bib`** — bibliography

#### Paper structure (TCSS)

| Section | Source material |
|---------|-----------------|
| Abstract | `Analysis/analysis.md` executive summary |
| Introduction | README §1, validity gap headline (39% BBQ hidden invalidity) |
| Related work | `references.bib`, README §19 |
| Method — Pentad | README §4, `pentad_generator.py` |
| Method — CDVA | README §6, position detection fix (§6.5) |
| Method — PAV framework | README §3, `Analysis/04_algebraic_validity.py` |
| Method — Scoring | `CPU_Only/scoring.py`, tau calibration note |
| Results — OSM | `Analysis/analysis.md` §2–6, `results/validity_gap_leaderboard.md` |
| Results — API | After Phase A completes |
| Results — Failure modes | `Analysis/02_failure_modes.py`, leaderboard matrix |
| Results — Statistical tests | `Analysis/05_statistical_tests.py` (McNemar, Cohen's h) |
| Limitations | Position detection fix, tau percentile fallback, GPT-OSS via OpenRouter proxy |
| Figures | `Analysis/outputs/figures/` (6 figures, 300 DPI) |

#### Critical rules for journal correctness

1. **CDVA analysis:** Always filter `position_fallback_used == False`
2. **MIRAGE-Full:** Only report for OSM models (API models have no CDVA — use MIRAGE-B only for API)
3. **Tau:** Disclose 75th-percentile calibration; dev-seed calibration is future work
4. **API routing/versions:** all four API models have an OpenRouter secondary fallback to the same model — API-1 `qwen3-next-80b-a3b` and API-2 `amazon-nova-2-lite` (AWS Bedrock → OpenRouter), API-3 `gemini-2.5-flash` (LinkAPI `geminicheap` → OpenRouter `google/gemini-2.5-flash` → MegaLLM; MegaLLM dropped to last after its gemini credits were exhausted mid-run), API-4 `mistral-medium` (Mistral → OpenRouter `mistralai/mistral-medium-3-5`). DeepSeek is generator + JSON-repair judge only (not evaluated), so no evaluated model is also a probe generator. Record the `route_used` column and disclose that `gemini-2.5-flash`, `mistral-medium-latest`, and the OpenRouter aliases track provider "latest" snapshots (pin dated versions for full reproducibility).
5. **Parse failures:** Qwen 1.15%, Phi 1.24% — exclude from MIRAGE-B via `success_flag=False`, not a validity signal
6. **Do not mix** dry-run or test results with production — production `run_id` is in parquet metadata

#### Figures for paper

| Fig | File | Caption idea |
|-----|------|--------------|
| 1 | `fig1_leaderboard_heatmap.png` | Failure mode rates by benchmark |
| 2 | `fig2_validity_gap_bars.png` | Native vs MIRAGE-Full with bootstrap CIs |
| 3 | `fig3_cdva_violin.png` | CDVA commutator distribution per model |
| 4 | `fig4_failure_stacked.png` | FM decomposition per model × benchmark |
| 5 | `fig5_commutator_by_axis.png` | Commutator by demographic axis |
| 6 | `fig6_measdefect_cdf.png` | Measurement defect CDF per model |

---

### Phase C — Optional Improvements (if time permits)

| Task | Why | How |
|------|-----|-----|
| Tag dev seeds + proper tau calibration | Stronger methods claim | Tag 50 seeds in pentad, run `GPU_CPU/cdva_calibration.py` |
| WinoBias predictive validity | Paper contribution #5 | Evaluate WinoBias with API+OSM, run `CPU_Only/predictive_validity.py` |
| Re-run Analysis with API results | Complete 8-model tables | After Phase A |
| Tau sensitivity analysis | Robustness check | Vary tau 50th–90th percentile, report MIRAGE-Full stability |

---

## 6. Important Warnings

| Do | Don't |
|----|-------|
| Run API on a VM with `tmux`/`nohup` | Run API on laptop without persistent session |
| Dry run before API production | Skip dry run |
| Use sequential API (default in code) | Add parallelism to API calls (hits rate limits) |
| Filter CDVA by `position_fallback_used=False` | Use all CDVA rows blindly |
| Report MIRAGE-Full for OSM only | Report MIRAGE-Full for API models |
| Resume with same `run_cpu_full.py` after crash | Delete `behavioral_results.parquet` |
| Keep `.env` local only | Commit `.env` to GitHub |

**GCP VM `audit` can be deleted** — all production GPU results are on GitHub and local clone.

---

## 7. Quick Reference Commands

```bash
# Repo root
cd Audit_Benchmark

# CPU dry run
cd Code/mirage && python Dry_Run/dry_run_cpu_only.py

# Full CPU pipeline (API + scoring)
cd Code/mirage && python run_cpu_full.py

# Scoring only (no API)
cd Code/mirage && python run_cpu_postprocess.py

# Full analysis
python Analysis/run_all.py

# Validity gap markdown table
cd Code/mirage && python -m CPU_Only.validity_gap_table

# Check row counts
python -c "import pandas as pd; print(len(pd.read_parquet('Code/mirage/results/behavioral_results.parquet')))"
```

---

## 8. Contact & Context

- GPU pipeline ran on GCP A100 40 GB (Jun 4–5, 2026)
- Results validated for journal submission to TCSS
- All critical bugs resolved; production data is clean
- API run is the main remaining execution task before paper can include all 8 models

For detailed GPU run history, bugs, and resume rules: **`Help/VM_progress.md`**
