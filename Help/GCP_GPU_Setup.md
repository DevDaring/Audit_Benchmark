# GCP GPU Setup — MIRAGE Project

Complete guide for running the MIRAGE GPU pipeline on **Google Cloud Compute Engine** with an **NVIDIA A100 40 GB** (`a2-highgpu-1g`). The pipeline auto-detects 40 GB VRAM and loads **one OSM model at a time** (sequential mode).

**Related docs:** [`Code/mirage/README.md`](../Code/mirage/README.md) · [`Help/Akash_VM_Setup.md`](Akash_VM_Setup.md) · [`Help/VM_progress.md`](VM_progress.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Create the VM](#3-create-the-vm)
4. [First Login and System Checks](#4-first-login-and-system-checks)
5. [Clone the Repository](#5-clone-the-repository)
6. [Python Environment and Package Installation](#6-python-environment-and-package-installation)
7. [Configure `.env`](#7-configure-env)
8. [Pre-Download HuggingFace Models](#8-pre-download-huggingface-models)
9. [Build or Restore the Pentad Dataset](#9-build-or-restore-the-pentad-dataset)
10. [Dry Run (Recommended)](#10-dry-run-recommended)
11. [Run the GPU Pipeline](#11-run-the-gpu-pipeline)
12. [Monitor Progress](#12-monitor-progress)
13. [Download Results](#13-download-results)
14. [Post-GPU Steps (Local or Same VM)](#14-post-gpu-steps-local-or-same-vm)
15. [Cost Estimate](#15-cost-estimate)
16. [Troubleshooting](#16-troubleshooting)
17. [Teardown](#17-teardown)

---

## 1. Overview

| Item | Value |
|---|---|
| **Recommended instance** | `a2-highgpu-1g` |
| **GPU** | 1× NVIDIA A100 **40 GB** |
| **vCPU / RAM** | 12 vCPUs / **85 GB** RAM |
| **OS** | Ubuntu 22.04 LTS (Deep Learning image with CUDA 12.4) |
| **Boot disk** | **≥ 200 GB** SSD (models + cache + results) |
| **VRAM strategy** | Sequential — one model at a time (~16 GB peak) |
| **Production pentad** | N = 596 seeds × 12 = **7,152 rows** |
| **Expected GPU time** | ~8–12 GPU-hours (full fresh run) |

**What runs on GCP:** OSM behavioral evaluation (4 models), CDVA activation patching, tau calibration.

**What can run elsewhere:** DeepSeek slot d/e regeneration, API model evaluation, scoring, figures (CPU + API keys).

---

## 2. Prerequisites

### GCP account

1. A GCP project with **billing enabled**.
2. **GPU quota** for `NVIDIA A100 GPUs` in your chosen region (default quota is often **0**).
   - Console → **IAM & Admin → Quotas** → filter `NVIDIA A100 GPUs` → request increase (usually 1 is enough).
3. `gcloud` CLI installed locally ([install guide](https://cloud.google.com/sdk/docs/install)).

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### HuggingFace access

- Accept the Llama 3.1 and Gemma 2 licenses on HuggingFace.
- Create a **read token** at https://huggingface.co/settings/tokens.

### API keys (for full pipeline)

Copy `Code/mirage/.env.example` → `.env`. GPU-only work needs at minimum `HUGGINGFACE_TOKEN`. Full MIRAGE also needs DeepSeek keys (slots d/e) and API keys for scoring — see [Section 7](#7-configure-env).

---

## 3. Create the VM

Replace `YOUR_PROJECT_ID`, `us-central1-a`, and `mirage-gpu` as needed.

### Option A — `gcloud` (recommended)

```bash
gcloud compute instances create mirage-gpu \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=common-cu124-ubuntu-2204-nvidia \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --restart-on-failure \
  --scopes=cloud-platform
```

Notes:

- **`a2-highgpu-1g`** already includes one A100 40 GB — do not add a separate `--accelerator` flag.
- **`TERMINATE`** is required for GPU VMs (live migration is not supported).
- Use **`pd-balanced`** or **`pd-ssd`** for faster model downloads.
- If the image family is unavailable in your region, try `ubuntu-2204-lts` and install NVIDIA drivers manually (see [Troubleshooting](#16-troubleshooting)).

### Option B — Cloud Console

1. **Compute Engine → VM instances → Create instance**
2. **Name:** `mirage-gpu`
3. **Region / zone:** e.g. `us-central1-a` (must have A100 quota)
4. **Machine configuration:** GPU → **A2** → **a2-highgpu-1g (1 GPU)**
5. **Boot disk:** Ubuntu 22.04, **200 GB**
6. **Advanced → Availability:** Terminate VM instance
7. Create

### Firewall (optional, for browser-based SSH only)

Default SSH via `gcloud compute ssh` works without opening port 22 to the world.

---

## 4. First Login and System Checks

```bash
gcloud compute ssh mirage-gpu --zone=us-central1-a
```

On the VM:

```bash
# GPU visible
nvidia-smi

# Expected: NVIDIA A100-SXM4-40GB (or similar), driver ≥ 525, CUDA 12.x

# Python version (note for flash-attn wheel — Section 6)
python3 --version

# Disk space (need ~100 GB free after OS)
df -h /
```

| Check | Expected |
|---|---|
| `nvidia-smi` | A100 40 GB, no errors |
| `python3 --version` | 3.10 or 3.12 |
| Free disk | ≥ 150 GB on `/` |

---

## 5. Clone the Repository

```bash
sudo apt-get update
sudo apt-get install -y git tmux htop

export MIRAGE_ROOT="$HOME/Audit_Benchmark"
git clone https://github.com/YOUR_ORG/Audit_Benchmark.git "$MIRAGE_ROOT"
cd "$MIRAGE_ROOT/Code/mirage"
```

Use your actual GitHub URL. For a private repo:

```bash
git clone https://YOUR_TOKEN@github.com/YOUR_ORG/Audit_Benchmark.git "$MIRAGE_ROOT"
```

---

## 6. Python Environment and Package Installation

MIRAGE requires **Linux x86_64**, **CUDA 12.4**, **flash-attention-2**, and pinned packages in `requirements.txt`. Use a **dedicated venv** (do not rely on system pip alone).

### 6.1 System build dependencies

```bash
sudo apt-get install -y \
  python3-dev python3-venv build-essential \
  curl wget pkg-config
```

### 6.2 Create virtualenv

```bash
cd "$MIRAGE_ROOT/Code/mirage"
python3 -m venv ~/mirage-venv
source ~/mirage-venv/bin/activate

python -m pip install --upgrade pip setuptools wheel packaging ninja
python --version   # remember this for flash-attn wheel (3.10 vs 3.12)
```

Add to `~/.bashrc` so SSH sessions auto-activate:

```bash
echo 'source ~/mirage-venv/bin/activate' >> ~/.bashrc
```

### 6.3 Install PyTorch (CUDA 12.4)

Matches `requirements.txt` (`torch==2.5.1`):

```bash
python -m pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 6.4 Install flash-attention (prebuilt wheel)

**Do not** `pip install flash-attn` from source unless the wheel fails — compile takes 45–60 minutes.

Pick the wheel matching your **Python** tag (`cp310` or `cp312`) and **torch 2.5**:

**Python 3.12:**

```bash
wget -O /tmp/flash_attn.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

python -m pip install --no-deps /tmp/flash_attn.whl
```

**Python 3.10:**

```bash
wget -O /tmp/flash_attn.whl \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python -m pip install --no-deps /tmp/flash_attn.whl
```

Verify:

```bash
python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"
```

If the URL 404s, browse [FlashAttention releases](https://github.com/Dao-AILab/flash-attention/releases) for `cu12torch2.5cxx11abiFALSE` matching your Python version.

### 6.5 Install remaining MIRAGE dependencies

```bash
cd "$MIRAGE_ROOT/Code/mirage"
python -m pip install -r requirements.txt
```

### 6.6 Full verification script

Run once after install:

```bash
python - <<'EOF'
import sys
checks = []

def ok(name, fn):
    try:
        fn()
        checks.append((name, True, ""))
    except Exception as e:
        checks.append((name, False, str(e)))

ok("torch+cuda", lambda: (
    __import__("torch").cuda.is_available() and print(__import__("torch").cuda.get_device_name(0))
))
ok("flash_attn", lambda: __import__("flash_attn"))
ok("transformers", lambda: __import__("transformers"))
ok("transformer_lens", lambda: __import__("transformer_lens"))
ok("nnsight", lambda: __import__("nnsight"))
ok("outlines", lambda: __import__("outlines"))
ok("pandas", lambda: __import__("pandas"))

for name, passed, err in checks:
    print(f"{'PASS' if passed else 'FAIL':4}  {name}" + (f"  ({err})" if err else ""))
sys.exit(0 if all(p for _, p, _ in checks) else 1)
EOF
```

All lines must show `PASS` before running production GPU work.

### 6.7 HuggingFace cache directory

Point model weights to a stable path (optional but recommended):

```bash
mkdir -p "$HOME/hf_cache"
export HF_HOME="$HOME/hf_cache"
echo "export HF_HOME=$HOME/hf_cache" >> ~/.bashrc
```

---

## 7. Configure `.env`

```bash
cd "$MIRAGE_ROOT/Code/mirage"
cp .env.example .env
nano .env   # or vim
```

| Variable | Required for GPU | Purpose |
|---|---|---|
| `HUGGINGFACE_TOKEN` | **Yes** | Llama 3.1, Gemma 2 (gated) |
| `DEEPSEEK_API_KEY_1/2` | For dataset regen | Pentad slots d/e |
| `AWS_ACCESS_KEY`, `AWS_SECRET_KEY` | Post-GPU | API models (Bedrock) |
| `GEMINI_API_KEY_*` | Post-GPU | API models |
| `MISTRAL_API_KEY1/2` | Post-GPU | API models |
| `OPENROUTER_API_KEY_*` | Post-GPU | Fallback routing |

**Never commit `.env`.** Strip Windows CRLF if edited on Windows:

```bash
sed -i 's/\r$//' .env
```

Export for the current session:

```bash
set -a && source .env && set +a
export HF_TOKEN="$HUGGINGFACE_TOKEN"
```

---

## 8. Pre-Download HuggingFace Models

Download all four OSM weights **before** the long GPU run (~35–45 GB total). This avoids mid-pipeline download failures.

```bash
cd "$MIRAGE_ROOT/Code/mirage"
source ~/mirage-venv/bin/activate
set -a && source .env && set +a

python - <<'EOF'
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
cache = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))

models = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-4-mini-instruct",
]
for repo in models:
    print(f"Downloading {repo} ...")
    snapshot_download(repo_id=repo, cache_dir=cache, token=token)
    print(f"  OK: {repo}")
print("All models cached.")
EOF
```

Or use the Akash helper (works on GCP too):

```bash
export HF_HOME="$HOME/hf_cache"
export STATE_DIR="$HOME/mirage-state"
mkdir -p "$STATE_DIR"
python "$MIRAGE_ROOT/akash/predownload_models.py"
```

---

## 9. Build or Restore the Pentad Dataset

### Option A — Copy existing production pentad (fastest)

If you already have a validated `pentad_dataset.parquet` from a prior run:

```bash
mkdir -p "$MIRAGE_ROOT/Code/mirage/Dataset/seeds"
# From your laptop:
# gcloud compute scp ./pentad_dataset.parquet mirage-gpu:~/Audit_Benchmark/Code/mirage/Dataset/seeds/ --zone=us-central1-a

python - <<'EOF'
import pandas as pd
from Dataset.validate_pentad import assert_production_ready
df = pd.read_parquet("Dataset/seeds/pentad_dataset.parquet")
assert_production_ready(df)
print("OK:", len(df), "rows")
EOF
```

Must pass with **7,152 rows** for production.

### Option B — Full rebuild on GCP

Run in `tmux` — slot d/e API generation can take many hours.

```bash
cd "$MIRAGE_ROOT/Code/mirage"
source ~/mirage-venv/bin/activate
set -a && source .env && set +a

tmux new -s mirage-dataset
```

Inside tmux:

```bash
# 1. Source datasets
python -c "
from Dataset.download_bbq import download_bbq, validate_bbq
from Dataset.download_crows_pairs import download_crows_pairs, validate_crows_pairs
from Dataset.download_stereoset import download_stereoset, validate_stereoset
from Dataset.download_winobias import download_winobias, validate_winobias
validate_bbq(download_bbq())
validate_crows_pairs(download_crows_pairs())
validate_stereoset(download_stereoset())
validate_winobias(download_winobias())
"

# 2. Sample seeds
python -c "
from Dataset.sample_seeds import sample_seeds, verify_seeds_integrity
main, dev = sample_seeds()
verify_seeds_integrity()
print(len(main), 'main,', len(dev), 'dev')
"

# 3. Build pentad
python run_dataset.py
python patch_slot_b_only.py

# 4. Regenerate DeepSeek slots d/e (resume-safe)
python regenerate_api_slots.py
# If interrupted:
python regenerate_api_slots.py --keep-checkpoint

# 5. Production gate
python -c "
import pandas as pd
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar
df = pd.read_parquet('Dataset/seeds/pentad_dataset.parquet')
validate_slot_b_grammar(df)
assert_production_ready(df)
print('OK:', len(df), 'rows')
"
```

Detach tmux: `Ctrl-b` then `d`.

---

## 10. Dry Run (Recommended)

Confirms GPU, flash-attention, and sequential loading before the full 7,152-row run.

```bash
cd "$MIRAGE_ROOT/Code/mirage"
source ~/mirage-venv/bin/activate
set -a && source .env && set +a

# CPU-only checks (no GPU)
python Dry_Run/dry_run_all.py --skip-gpu

# Full dry run (2 seeds, sequential model load/unload)
python Dry_Run/dry_run_all.py
```

Look for `DRYRUN_OK` in logs / markers. Fix any `FAIL` before proceeding.

---

## 11. Run the GPU Pipeline

### 11.1 Sequential loading (automatic on 40 GB)

On A100 40 GB, `run_gpu_pipeline.py` detects VRAM < 48 GB and runs:

```
For each model (Llama → Qwen → Gemma → Phi):
  load → behavioral (det + FM4 variance) → CDVA → unload
Then: tau calibration on CPU
```

No env vars needed on 40 GB. Optional overrides:

```bash
export MIRAGE_SEQUENTIAL_MODELS=1      # force sequential
export MIRAGE_EVAL_BATCH_SIZE=4        # default on 40 GB; use 2 if OOM
```

### 11.2 Start production run

Use **tmux** so SSH disconnect does not kill the job:

```bash
cd "$MIRAGE_ROOT/Code/mirage"
source ~/mirage-venv/bin/activate
set -a && source .env && set +a
export HF_HOME="$HOME/hf_cache"

tmux new -s mirage-gpu
python GPU_CPU/run_gpu_pipeline.py 2>&1 | tee ~/mirage-gpu.log
```

Re-attach: `tmux attach -t mirage-gpu`

### 11.3 Resume after interruption

The pipeline is **checkpoint-resume safe**. Re-run the same command; completed rows in `results/behavioral_results.parquet` and `results/cdva_results.parquet` are skipped.

```bash
python GPU_CPU/run_gpu_pipeline.py
```

**Do not** delete partial parquets unless you intend a full rerun.

### 11.4 Expected log lines

```
Sequential model loading enabled (GPU VRAM=40.0 GB; one model at a time, peak ~16 GB).
Step 1–3 (model 1/4): llama-3.1-8b-instruct — load → behavioral → CDVA → unload ...
Model 1/4 (llama-3.1-8b-instruct) complete ...
...
=== GPU PIPELINE COMPLETE ===
```

---

## 12. Monitor Progress

### GPU utilization

```bash
watch -n 5 nvidia-smi
```

During active inference you should see GPU util 70–100% and ~16–24 GB VRAM used (one model).

### Row counts

```bash
cd "$MIRAGE_ROOT/Code/mirage"
python - <<'EOF'
import pandas as pd
from pathlib import Path

beh = Path("results/behavioral_results.parquet")
cdva = Path("results/cdva_results.parquet")
if beh.exists():
    df = pd.read_parquet(beh)
    print("behavioral:", len(df), "rows | models:", df["model_name"].value_counts().to_dict())
if cdva.exists():
    df = pd.read_parquet(cdva)
    print("cdva:", len(df), "rows | models:", df["model_name"].value_counts().to_dict())
EOF
```

### Tail log

```bash
tail -f ~/mirage-gpu.log
```

### Production targets (approximate)

| Model | Behavioral rows (det + 5× FM4 on slot-a) | CDVA seeds |
|---|---|---|
| Each of 4 OSMs | varies; total ~20k+ clean rows | 596 seeds × 10 pairs |

See [`Help/VM_progress.md`](VM_progress.md) for stage markers and ETAs.

---

## 13. Download Results

From your **local machine**:

```bash
ZONE=us-central1-a
VM=mirage-gpu

gcloud compute scp --recurse \
  "$VM:~/Audit_Benchmark/Code/mirage/results" \
  ./mirage-results-from-gcp/ \
  --zone="$ZONE"

gcloud compute scp \
  "$VM:~/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet" \
  ./mirage-results-from-gcp/ \
  --zone="$ZONE"
```

Key files:

| File | Content |
|---|---|
| `results/behavioral_results.parquet` | OSM behavioral outputs |
| `results/cdva_results.parquet` | CDVA patching scores |
| `results/logs/` | Run logs |
| `Dataset/seeds/pentad_dataset.parquet` | Input pentad |

---

## 14. Post-GPU Steps (Local or Same VM)

GPU work is only steps 5–7 of the full MIRAGE pipeline. After downloading results (or on the same VM without GPU):

```bash
cd "$MIRAGE_ROOT/Code/mirage"
source ~/mirage-venv/bin/activate
set -a && source .env && set +a

# API behavioral (4 API models — no GPU)
python -c "
import pandas as pd
from CPU_Only.api_behavioral import run_api_behavioral
from config import RESULTS_DIR, SEEDS_DIR
run_api_behavioral(pd.read_parquet(SEEDS_DIR / 'pentad_dataset.parquet'), run_id='main_run')
"

# Scoring, leaderboard, figures — see Code/mirage/README.md Section 12
```

You can stop or delete the GPU VM after OSM behavioral + CDVA complete to save cost, and finish API/scoring on a cheap CPU instance or your laptop.

---

## 15. Cost Estimate

Prices vary by region; approximate **on-demand** US pricing:

| Resource | ~Rate | Full run (~10 GPU-h) |
|---|---|---|
| `a2-highgpu-1g` | ~$3.50–4.00 / hr | ~$35–40 |
| 200 GB boot disk | ~$0.02 / GB-mo | negligible for days |
| Egress (download results) | ~$0.12 / GB | ~$1–5 |

**Tips to reduce cost:**

- Use **Spot** VMs if you can tolerate preemption (pipeline resumes from parquet).
- **Stop** the VM when not running (`gcloud compute instances stop mirage-gpu`).
- Delete VM after downloading results.
- Pre-download models once; avoid repeated create/destroy without cache.

---

## 16. Troubleshooting

### GPU quota exceeded

```
Quota 'NVIDIA_A100_GPUS' exceeded
```

Request quota increase in Console → Quotas, or try another region (`us-east4`, `europe-west4`).

### `nvidia-smi` not found

Install drivers on plain Ubuntu:

```bash
sudo apt-get install -y linux-headers-$(uname -r)
# Use Google's install script or CUDA keyring — prefer switching to
# deeplearning-platform-release image and recreating the VM.
```

Recreating with the Deep Learning image is usually faster.

### `flash_attn` import fails

| Symptom | Fix |
|---|---|
| Wrong Python tag | Re-download wheel with `cp310` vs `cp312` matching `python --version` |
| Wrong torch | Must be `torch==2.5.1+cu124`; reinstall before flash-attn |
| `cxx11abiTRUE` wheel | Use **`cxx11abiFALSE`** wheels only |

### CUDA OOM during behavioral

```bash
export MIRAGE_EVAL_BATCH_SIZE=2   # or 1
python GPU_CPU/run_gpu_pipeline.py
```

### CUDA OOM during CDVA (TransformerLens)

CDVA already unloads the HF copy when VRAM is tight. If OOM persists:

- Ensure sequential mode is active (only one model loaded).
- Restart Python process to clear fragmented VRAM, then resume.

### HuggingFace 401 on Llama / Gemma

- Token set in `.env` as `HUGGINGFACE_TOKEN`
- Accept model licenses on huggingface.co
- `export HF_TOKEN="$HUGGINGFACE_TOKEN"`

### `load_all_osm_models` RuntimeError on 40 GB

You called `load_all_osm_models()` directly — it refuses to load all four on <42 GB. Use:

```bash
python GPU_CPU/run_gpu_pipeline.py
```

which uses sequential loading automatically.

### Pentad gate fails

```
assert_production_ready() failed
```

- Need **7,152** rows with valid slots a–e.
- Do not start GPU on partial pentad.
- See [`Help/Akash_VM_Setup.md`](Akash_VM_Setup.md) Section 13 for validation gates.

### SSH disconnect stopped the pipeline

Always run inside **tmux** or **systemd** (Section 11.2). Re-run the same command to resume.

---

## 17. Teardown

```bash
# Stop billing compute (keeps disk)
gcloud compute instances stop mirage-gpu --zone=us-central1-a

# Delete VM entirely
gcloud compute instances delete mirage-gpu --zone=us-central1-a
```

Download `results/` and `pentad_dataset.parquet` **before** delete if the boot disk is not snapshotted.

---

## Quick Reference

```bash
# Login
gcloud compute ssh mirage-gpu --zone=us-central1-a

# Activate env
source ~/mirage-venv/bin/activate
cd ~/Audit_Benchmark/Code/mirage
set -a && source .env && set +a
export HF_HOME=~/hf_cache

# Production GPU (tmux recommended)
python GPU_CPU/run_gpu_pipeline.py 2>&1 | tee ~/mirage-gpu.log
```

| Setting | A100 40 GB default |
|---|---|
| Model loading | Sequential (auto) |
| `MIRAGE_EVAL_BATCH_SIZE` | 4 |
| Peak VRAM | ~16–24 GB |
| Host RAM | 85 GB (sufficient) |
