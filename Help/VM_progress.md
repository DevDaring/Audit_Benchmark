# VM Progress — MIRAGE GCP GPU Pipeline

Reference for monitoring, stage markers, ETAs, and safe resume on the production GCP A100 VM.

See also: **`Help/GCP_GPU_Setup.md`** (VM create, install, package setup) and **`Code/mirage/README.md`** (full research documentation).

---

## Current Production VM (GCP A100 40 GB)

| Field | Value |
|---|---|
| Project | `solar-nation-470113-r4` |
| Instance name | `audit` |
| Zone | `us-central1-f` |
| SSH key | `C:\Users\Debz\.ssh\id_rsa_gcp` |
| Username | `koushikdeb2009` |
| Repo | `/home/koushikdeb2009/Audit_Benchmark` |
| HF cache | `/home/koushikdeb2009/hf_cache` |
| State dir | `/home/koushikdeb2009/mirage-state` |
| Log | `~/mirage_prod.log` |

Connect:

```bash
gcloud compute ssh koushikdeb2009@audit \
  --zone=us-central1-f \
  --project=solar-nation-470113-r4 \
  --ssh-key-file=C:\Users\Debz\.ssh\id_rsa_gcp
```

---

## Pipeline Stages

```
DATASET_OK        → pentad validated (7,152 rows, assert_production_ready)
BEHAVIORAL_OK     → 4 models × 10,132 behavioral rows each
CDVA_OK           → 4 models × 5,960 CDVA pairs each
TAU_CALIB_OK      → tau_calibration.json written
PIPELINE_COMPLETE → final sentinel
```

### Stage detail

| Stage | What runs | Expected output |
|---|---|---|
| Pre-download | Models cached in hf_cache | ~42 GB in hf_cache |
| Dataset | `run_dataset.py` + `regenerate_api_slots.py` | 7,152 rows, assert_production_ready passes |
| GPU Step 1 | Load 1 OSM model at a time (`MIRAGE_SEQUENTIAL_MODELS=1`) | ~14–16 GB VRAM per model |
| GPU Step 2 | Behavioral eval (det + 5 variance passes per model) | `behavioral_results.parquet` |
| GPU Step 3 | CDVA patching per model | `cdva_results.parquet` |
| GPU Step 4 | Tau calibration | `tau_calibration.json` |

**Production audit set:** N = **596** seeds × 12 slots = **7,152** rows  
(BBQ 254, CrowS-Pairs 181, StereoSet 161; 22 StereoSet seeds excluded)

---

## Current Run Status (last updated: Jun 4, 2026 — 11:21 IST)

**Active pipeline PID 29226** — CDVA rerun with position-detection fix (all 4 models, behavioral already complete for Llama/Qwen/Gemma).

| Model | Behavioral | CDVA | Notes |
|---|---|---|---|
| llama-3.1-8b-instruct | DONE (10,132 rows, 0 failures) | RERUNNING (position fix active) | TransformerLens |
| qwen2.5-7b-instruct | DONE (10,132 rows, 122 parse fails = 1.20%) | RERUNNING after nnsight proxy fix | nnsight |
| gemma-2-2b-it | DONE (10,132 rows, 1 parse fail = 0.01%) | RERUNNING (position fix active) | TransformerLens |
| phi-4-mini-instruct | NOT STARTED (will run with batch_size=4) | NOT STARTED | nnsight |

**ETA to completion from 11:21 IST Jun 4:**
- All 4 × CDVA (~60 min each): ~4 hours
- Phi-4-mini behavioral (7,152 prompts, batch=4): ~3–3.5 hours (runs in parallel with Llama/Qwen/Gemma CDVA since behavioral is per-model before CDVA)
- Phi CDVA: ~60 min
- **Total: ~5–6 hours = completion by 16:30–17:30 IST Jun 4**

---

## Known Bugs Fixed in This Run

| Bug | Symptom | Fix (commit) |
|---|---|---|
| Gemma-2 CDVA device mismatch | `HookedTransformer.from_pretrained` raises "Expected all tensors on same device" before cache line is reached; every pair re-attempts conversion | Temporarily move HF model to CPU before TL conversion, then move TL model + deep-scan non-registered attributes to GPU. (commit `0f7a1ba`) |
| Qwen/Phi CDVA `AttributeError` — nnsight proxy chain | `_nnsight_layer_proxies` built incorrect proxy path (`nn_model.model.model.layers`); previous "fix" to `hf_model.model.layers` gave raw `nn.Module` objects with no `.output` attribute | **Root fix (commit `6fc63db`):** access layers INSIDE each `nn_model.trace()` context via `nn_model.model.layers[i]` (nnsight proxy objects); `lm_head` captured via `nn_model.lm_head.output.save()` |
| Phi-4-mini behavioral batch_size=1 | `use_constrained_single=True` for nnsight models forced outlines single-prompt path (~12 s/prompt); 7,152 det-pass prompts would take ~32 hours | Removed `use_constrained_single` check; batch generation (batch=4, ~7 s/batch) reduces det-pass to ~3 hours (commit `6fc63db`) |
| CDVA position detection failure for multi-word swap tokens | Swap tokens stored with underscores (`a_girl`, `middle_aged`, `a_trailer_park`); tokenizer never produces underscore-delimited tokens, so 53% of pairs fell back to `pos=1` (BOS prefix) — wrong position — producing trivially-zero delta_logit (91.5% zeros in fallback rows) | **Root fix (commit `fa47626`):** normalise underscores to spaces, then apply 3-pass char-level search: (1) single-token substring, (2) full-phrase in concatenated decoded string with char→token mapping, (3) last-word heuristic. Fallback rate drops from ~53% to < 10%. All CDVA results wiped and rerun with the fix. |
| Behavioral resume slow | `_completed_keys_from` used `iterrows()` on up to 171k rows — 5–15 min between models | Vectorised with `isin` and set comprehensions; resume overhead now < 5 s. |
| Pentad WinoBias contamination | `ValueError: WinoBias rows in pentad — must be held out` | `pentad_generator.py` now auto-excludes WinoBias seeds. |

---

## On-VM Quick Checks

```bash
# Process
pgrep -fa run_gpu_pipeline

# GPU
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Log tail
tail -20 ~/mirage_prod.log

# Behavioral rows per model
python3 -c "
import pandas as pd
b = pd.read_parquet('Code/mirage/results/behavioral_results.parquet')
for m, g in b.groupby('model_name'):
    ok = int(g.success_flag.sum()); fail = int((~g.success_flag).sum())
    print(m, 'rows=%d ok=%d fail=%d' % (len(g), ok, fail))
"

# CDVA rows per model
python3 -c "
import pandas as pd
c = pd.read_parquet('Code/mirage/results/cdva_results.parquet')
for m, g in c.groupby('model_name'):
    ok = int(g.success_flag.sum()); fail = int((~g.success_flag).sum())
    print(m, 'rows=%d ok=%d fail=%d' % (len(g), ok, fail))
"
```

---

## Start / Restart Pipeline

```bash
cd /home/koushikdeb2009/Audit_Benchmark/Code/mirage
export HF_HUB_CACHE=/home/koushikdeb2009/hf_cache \
       HF_HOME=/home/koushikdeb2009/hf_cache \
       MIRAGE_SEQUENTIAL_MODELS=1 \
       MIRAGE_EVAL_BATCH_SIZE=4 \
       STATE_DIR=/home/koushikdeb2009/mirage-state
set -a && source .env && set +a
export HF_TOKEN=${HUGGINGFACE_TOKEN} MIRAGE_SEQUENTIAL_MODELS=1 MIRAGE_EVAL_BATCH_SIZE=4
nohup python3 GPU_CPU/run_gpu_pipeline.py >> ~/mirage_prod.log 2>&1 &
echo $!
```

`MIRAGE_SEQUENTIAL_MODELS=1` must be exported **after** sourcing `.env` to prevent `.env` overriding it.

---

## Safe Resume Rules

| Situation | Action |
|---|---|
| CDVA parquet has failed rows for a model | Run cleanup script to remove failed rows for that model only; restart pipeline |
| CDVA parquet has all-zero delta_logit values | Check `position_fallback_used` distribution — if > 10%, the position-detection fix (commit `fa47626`) is not deployed; pull latest and wipe+rerun CDVA |
| TL conversion fails for Gemma-2 | The CPU-first fix in `utils_attention.py` (commit `0f7a1ba`) resolves this |
| nnsight `AttributeError` for Qwen/Phi | Layer access must be inside trace via `nn_model.model.layers[i]`; commit `6fc63db` resolves this |
| Behavioral parquet has stale rows | Resume logic in `osm_behavioral.py` skips completed (prompt_id, model, sample_index) triples automatically |
| Pipeline killed mid-CDVA | Restart without any cleanup — CDVA resume skips seeds whose (seed_id, model_name) are in the parquet with `success_flag=True` |
| Phi behavioral very slow (batch_size=1 in log) | Check that commit `6fc63db` is deployed; `use_constrained_single` must be `False` |

**CDVA analysis filter (mandatory for paper):** only use rows where `success_flag=True AND position_fallback_used=False`. The `position_fallback_used=True` rows have `pos_a = pos_b = 1` (wrong position) and produce trivially-zero delta_logit in 91.5% of cases — they convey no information about model bias.

**Never** run two pipeline instances simultaneously on the same parquet files.

---

## Expected Runtime (A100 40 GB, sequential loading)

| Phase | Duration | Notes |
|---|---|---|
| Behavioral per model | ~35–50 min | 10,132 rows, batch=4, flash_attn |
| CDVA per model (TransformerLens) | ~60 min | 596 seeds × 10 pairs; ~10 seeds/min |
| CDVA per model (nnsight) | ~60 min | Qwen, Phi-4-mini |
| Tau calibration | ~5 min | CPU-only |
| **Full GPU phase (4 models)** | **~7–8 hr** | Sequential loading overhead included |

TL conversion (CPU→GPU) for Gemma-2: ~20 s (one-time, cached thereafter).

---

## Post-Completion

When pipeline finishes:

1. Download `results/behavioral_results.parquet`, `results/cdva_results.parquet`, `results/tau_calibration.json`
2. Download `Dataset/seeds/pentad_dataset.parquet` and `pentad_manifest.json`
3. Run `CPU_Only/` scoring locally
4. Run `CPU_Only/leaderboard.py` and `CPU_Only/validity_gap_table.py`
5. Run `CPU_Only/predictive_validity.py`
6. Stop the GCP instance to stop billing: `gcloud compute instances stop audit --zone=us-central1-f`

---

## Deploy Code Fixes to Running VM

```bash
# SCP a specific file
gcloud compute scp "local/path/file.py" koushikdeb2009@audit:/remote/path/ \
  --zone=us-central1-f --project=solar-nation-470113-r4 \
  --ssh-key-file=C:\Users\Debz\.ssh\id_rsa_gcp

# Pull latest from GitHub on VM
gcloud compute ssh koushikdeb2009@audit --zone=us-central1-f \
  --project=solar-nation-470113-r4 \
  --ssh-key-file=C:\Users\Debz\.ssh\id_rsa_gcp \
  --command "cd /home/koushikdeb2009/Audit_Benchmark && git pull"
```

After updating code: clear `__pycache__` for changed modules, kill the running pipeline if it uses the old code, then restart.
