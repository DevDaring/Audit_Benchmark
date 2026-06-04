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

## Current Run Status (last updated: Jun 4, 2026)

| Model | Behavioral | CDVA | Notes |
|---|---|---|---|
| llama-3.1-8b-instruct | DONE (10,132 rows, 0 failures) | DONE (5,960/5,960 success) | TransformerLens |
| qwen2.5-7b-instruct | DONE (10,132 rows, 174 parse fails = 1.7%) | DONE (5,955/5,960 — 5 pairs excluded) | nnsight |
| gemma-2-2b-it | DONE (10,132 rows, 1 parse fail) | IN PROGRESS (~125/596 seeds as of 03:02 UTC) | TransformerLens |
| phi-4-mini-instruct | NOT STARTED | NOT STARTED | nnsight |

**ETA to completion:** ~2.5 hours from 03:08 UTC Jun 4 = ~05:30 UTC (11:00 AM IST)

---

## Known Bugs Fixed in This Run

| Bug | Symptom | Fix (commit) |
|---|---|---|
| Gemma-2 CDVA device mismatch | `HookedTransformer.from_pretrained` raises "Expected all tensors on same device" before cache line is reached; every pair re-attempts conversion | Temporarily move HF model to CPU before TL conversion, then move TL model + deep-scan non-registered attributes to GPU. (commit `0f7a1ba`) |
| Qwen/Phi CDVA AttributeError | `_nnsight_layer_proxies` built proxy chains (`nn_model.model.model.layers`) causing `'Qwen2Model' object has no attribute 'model'` | Changed to use actual HF module references: `inner.layers` and `hf_model.lm_head`. (commit `e2eab12`) |
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
| TL conversion fails for Gemma-2 | The CPU-first fix in `utils_attention.py` (commit `0f7a1ba`) resolves this |
| nnsight AttributeError for Qwen/Phi | The HF module ref fix in `utils_attention.py` (commit `e2eab12`) resolves this |
| Behavioral parquet has stale rows | Resume logic in `osm_behavioral.py` skips completed (prompt_id, model, sample_index) triples automatically |
| Pipeline killed mid-CDVA | Restart without any cleanup — CDVA resume skips seeds whose (seed_id, model_name) are in the parquet with `success_flag=True` |

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
