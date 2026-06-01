# Akash GPU VM Setup — MIRAGE Project
### Complete Field Guide: Deployment, Persistent Storage, Pipeline Checkpoints, Dataset Validation, and GPU Execution

*Production reference for MIRAGE on Akash Network. Covers deployment, package install,
persistent volumes, checkpoint markers, pentad validation gates, and GPU pipeline operation.*

---

## Table of Contents
1. [What Akash Is](#1-what-akash-is)
2. [Prerequisites](#2-prerequisites)
3. [Creating a Deployment via the Console API](#3-creating-a-deployment-via-the-console-api)
4. [SDL Configuration — Production (Persistent /data)](#4-sdl-configuration--production-persistent-data)
5. [Container Startup Sequence](#5-container-startup-sequence)
6. [Installing Python Packages — The Right Way](#6-installing-python-packages--the-right-way)
7. [Flash Attention — Compatibility and Installation](#7-flash-attention--compatibility-and-installation)
8. [Persistent Storage — Required for Production](#8-persistent-storage--required-for-production)
9. [Memory Management — cgroup Limits vs free -h](#9-memory-management--cgroup-limits-vs-free--h)
10. [Pre-Downloading All Models Before Running Code](#10-pre-downloading-all-models-before-running-code)
11. [Uploading .env to the VM](#11-uploading-env-to-the-vm)
12. [Production Pipeline — Markers, Supervisor, GPU Work](#12-production-pipeline--markers-supervisor-gpu-work)
13. [Dataset Validation Gates (Pentad Integrity)](#13-dataset-validation-gates-pentad-integrity)
14. [DeepSeek API Slots (d/e) Regeneration](#14-deepseek-api-slots-de-regeneration)
15. [GPU Pipeline Guards & Stale Result Prevention](#15-gpu-pipeline-guards--stale-result-prevention)
16. [Reset Protocol — Resume from Last Good Stage](#16-reset-protocol--resume-from-last-good-stage)
17. [Monitoring & Health Checks](#17-monitoring--health-checks)
18. [Post-GPU: CPU-Only Scoring (Local)](#18-post-gpu-cpu-only-scoring-local)
19. [Scripts Reference](#19-scripts-reference)
20. [Troubleshooting Index](#20-troubleshooting-index)
21. [Cost Estimate](#21-cost-estimate)

**Related:** [`Help/VM_progress.md`](VM_progress.md) — stage markers, monitoring scripts, ETAs, safe resume.

---

## 1. What Akash Is

Akash Network is a decentralised cloud marketplace. Providers (GPU node operators) bid on your deployment request. You accept the cheapest bid and pay in AKT (Akash token). The billing is per-block (~6 seconds per block, ~$1.25–5/hr for an A100).

Key difference from AWS: **you interact with the Akash Console REST API** rather than a VM console. Scripts in `akash/` automate the full flow.

---

## 2. Prerequisites

| Requirement | Detail |
|---|---|
| Akash Console API key | From `https://console.akash.network/` → Settings → API Keys |
| AKT wallet balance | Must be funded; billing is on-chain per block |
| `.env` file | `Code/mirage/.env` with all API keys filled in |
| GitHub repo (public or with token) | Container clones from GitHub on every boot |
| `paramiko` installed locally | `pip install paramiko` — for SSH helper scripts |

---

## 3. Creating a Deployment via the Console API

### API Flow (implemented in `akash/_deploy_mirage.py`)

```
POST /v1/deployments          → returns { dseq, manifest }
GET  /v1/bids?dseq=...        → poll until bids appear (up to 5 min)
POST /v1/leases               → { manifest, leases: [{dseq, gseq, oseq, provider}] }
GET  /v1/deployments/{dseq}   → poll until forwarded SSH port appears
```

### Issue 1: No bids received (most common first-timer problem)

**Symptoms:** Polling `GET /v1/bids?dseq=...` returns empty list indefinitely.

**Causes and fixes (in order of impact):**

| Cause | Fix |
|---|---|
| Bid price too low | Raise `amount` in the SDL's pricing block. We started at 10 000 µUAKT, had to go to **10 000 000 µUAKT** before bids appeared. |
| `signedBy` or `host` attributes set | Remove them entirely from the SDL. These restrict to specific providers and most providers don't match. |
| GPU `ram:` constraint set | Remove the `ram:` sub-field from the GPU block. Providers report VRAM in different ways; constraints often fail to match. |
| Too restrictive GPU model string | Use `nvidia:` with no sub-attributes, or `model: a100` with no RAM. Avoid `model: a100-80gb`. |
| `attributes.host: akash` | Remove this. Not needed and blocks most providers. |

**Working SDL placement block (the minimal version that actually got bids):**
```yaml
placement:
  westcoast:
    pricing:
      mirage:
        denom: uakt
        amount: 10000000
```

**Working GPU block:**
```yaml
gpu:
  units: 1
  attributes:
    vendor:
      nvidia:
```

### Issue 2: Lease creation fails with `manifest undefined`

**Symptom:** `POST /v1/leases` returns `{"code":"invalid_type","expected":"string","received":"undefined","path":["manifest"]}`.

**Cause:** The `manifest` string must be extracted from the **`POST /v1/deployments` response body** (`data.manifest`) and passed verbatim to `POST /v1/leases`. It is NOT the SDL text itself.

**Fix:**
```python
# Step 1: create deployment
r = requests.post(f"{BASE}/v1/deployments", json={"sdl": sdl_text}, headers=H)
dseq = r.json()["data"]["dseq"]
manifest_blob = r.json()["data"]["manifest"]   # <-- save this

# Step 2: create lease (after bids)
requests.post(f"{BASE}/v1/leases", json={
    "manifest": manifest_blob,                  # <-- pass the blob here, NOT the SDL
    "leases": [{"dseq": dseq, "gseq": 1, "oseq": 1, "provider": chosen_provider}]
}, headers=H)
```

### Issue 3: SSH port not appearing after lease accepted

**Cause:** Container is still booting (running apt-get, git clone, etc.). Takes 1–3 minutes.

**Fix:** Poll `GET /v1/deployments/{dseq}` every 10 seconds and wait for `services.mirage.uris` or the forwarded port list to be populated.

---

## 4. SDL Configuration — Production (Persistent /data)

The production SDL (in `akash/_deploy_mirage.py`) uses a **persistent `/data` volume** so venv, HF cache, state markers, and logs survive container evictions. Use this as the canonical template.

```yaml
---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
      - HF_HOME=/data/hf_cache
      - HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - PIP_CACHE_DIR=/data/pip_cache
      - XDG_CACHE_HOME=/data/cache
      - VENV=/data/venv
      - STATE_DIR=/data/state
      - REPO_DIR=/data/Audit_Benchmark
    command:
      - bash
      - -c
      - |
        apt-get update -qq && apt-get install -y git curl tmux openssh-server wget python3-venv python3-dev build-essential > /dev/null 2>&1
        rm -rf /var/lib/apt/lists/*
        echo "root:MirageVM2026!" | chpasswd
        mkdir -p /run/sshd
        sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
        sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
        /usr/sbin/sshd
        mkdir -p /data/logs /data/state /data/hf_cache /data/pip_cache /data/cache /workspace
        if [ ! -d /data/Audit_Benchmark/.git ]; then
          git clone https://github.com/DevDaring/Audit_Benchmark.git /data/Audit_Benchmark 2>&1 || true
        else
          git -C /data/Audit_Benchmark pull --ff-only 2>&1 || true
        fi
        echo "VM_READY $(date -u +%FT%TZ)" > /workspace/vm_ready.txt
        nohup bash /data/Audit_Benchmark/akash/watchdog.sh >> /data/logs/watchdog.log 2>&1 &
        nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
        tail -f /dev/null
    expose:
      - port: 22
        as: 22
        to:
          - global: true
    params:
      storage:
        data:
          mount: /data
          readOnly: false

profiles:
  compute:
    mirage:
      resources:
        cpu:
          units: 4
        memory:
          size: 64Gi
        storage:
          - size: 30Gi              # small ephemeral root — last to be disk-evicted
          - name: data
            size: 120Gi
            attributes:
              persistent: true
              class: beta3           # NVMe persistent SSD
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
                - model: rtx4090
                - model: rtx3090
                - model: a10
                - model: l4
                - model: a40
                - model: a6000
                - model: l40
                - model: l40s
                - model: a100

  placement:
    akash:
      pricing:
        mirage:
          denom: uakt
          amount: 10000000

deployment:
  mirage:
    akash:
      profile: mirage
      count: 1
```

**Resource guidelines:**

| Resource | Value | Why |
|---|---|---|
| `memory: 64Gi` | Mandatory | Model loading peaks need ~14 GB CPU RAM per 7–8B model; 64 GiB avoids cgroup OOM (see §9). |
| Ephemeral root `30Gi` | Small footprint | Keeps the container off the provider's disk-eviction list longer than a 200 GiB root. |
| Persistent `/data` `120Gi` | beta3/NVMe | Holds venv (~5 GB), HF cache (~50 GB), pentad, results, state markers. Survives evictions. |
| `runtime` CUDA image | Not `devel` | Smaller imagefs footprint on ephemeral root. |
| Wide GPU list | 24–80 GB | More bids; calmer providers; A100 preferred but not required. |
| `amount: 10000000` µUAKT | Bid price | Raise if no bids appear within 5 min. |

**Paths on the VM:**

```
/data/
  Audit_Benchmark/          ← git repo (cloned to persistent volume)
  venv/                     ← Python venv (survives eviction)
  hf_cache/                 ← HuggingFace model cache
  state/                    ← checkpoint markers (INSTALL_OK, DATASET_OK, …)
  logs/                     ← pipeline, install, watchdog logs
  .env                      ← uploaded via SFTP (not in git)
```

**Legacy note:** Early deployments used `/workspace` on ephemeral-only storage. Do not use that pattern for production — packages and markers are lost on every eviction.

---

## 5. Container Startup Sequence

After lease acceptance, the container runs the startup command:

1. `apt-get install` — git, curl, tmux, openssh-server, python3-venv, build-essential
2. Set root password and start SSH daemon
3. Create `/data/logs`, `/data/state`, cache directories
4. `git clone` into `/data/Audit_Benchmark` (persistent). **`git pull` is opt-in only** — set `MIRAGE_GIT_PULL=1` in the SDL env to pull on boot; default is off so uploaded hotfixes are not overwritten by `main`.
5. Launch `watchdog.sh` — logs cgroup memory, disk, GPU every 10 s to `/data/logs/watchdog.log`
6. Launch `supervise_pipeline.sh` — checkpoint-driven pipeline supervisor
7. `tail -f /dev/null` — keeps PID 1 alive

**SSH is available ~60–90 seconds after lease creation.**

The supervisor waits up to 600 s for `/data/.env` (uploaded via SFTP by `_deploy_mirage.py`), exports `HF_TOKEN` (stripping Windows CRLF), then runs `_full_pipeline.py`. On crash or eviction it retries automatically, skipping completed marker steps.

---

## 6. Installing Python Packages — The Right Way

### Issue 4: `python3 -m pip` broken after in-place upgrade (critical)

**What goes wrong with the naive approach:**
```bash
# BAD — do NOT do this
apt-get install python3-pip        # installs Ubuntu's patched pip
python3 -m pip install --upgrade pip  # this BREAKS python3 -m pip
```

**Why:** Ubuntu 22.04's system `python3-pip` is a patched version. When you upgrade pip in-place, the pip module is replaced but system Python's module discovery breaks. Result: `python3 -m pip` silently fails with `No module named pip`.

**The correct approach:** Bootstrap pip from `get-pip.py` FIRST:
```bash
# GOOD — always start with this
apt-get install -y python3-dev       # dev headers, NOT python3-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --quiet
python3 -m pip --version             # must succeed before proceeding
python3 -m pip install setuptools wheel packaging ninja
```

### Issue 5: Packages installed but not importable

**Symptom:** `pip install torch` reports success, but `python3 -c "import torch"` gives `ModuleNotFoundError`.

**Fix:** Always use `python3 -m pip install` (not `pip3 install`). Verify:
```bash
python3 -c "import sys; print(sys.executable, sys.path)"
python3 -m pip show torch | grep Location
```

Both should point to the same location.

### Issue 6: `transformer_lens.__version__` AttributeError breaks install script

**Symptom:** `install.sh` has `set -euo pipefail`. A verification line like:
```bash
python3 -c "import transformer_lens; print(transformer_lens.__version__)"
```
fails because `transformer_lens==2.18.0` has **no `__version__` attribute**, causing `set -e` to abort.

**Fix:**
```bash
# GOOD — use getattr with fallback
python3 -c "import transformer_lens; ver = getattr(transformer_lens, '__version__', 'installed'); print('transformer_lens', ver)"
```

### Correct install order for MIRAGE

```bash
# 1. Bootstrap pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
python3 -m pip install setuptools wheel packaging ninja

# 2. PyTorch 2.6.0 + CUDA 12.4
python3 -m pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Flash Attention (see §7 for full details)
python3 -m pip install "https://github.com/Dao-AILab/flash-attention/releases/download/\
v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    --no-deps

# 4. TransformerLens (pin v2 — our code uses v2 API)
python3 -m pip install "transformer_lens==2.18.0"

# 5. nnsight
python3 -m pip install "nnsight==0.7.0"

# 6. HuggingFace stack
python3 -m pip install "transformers>=4.47.0" "accelerate>=0.34.0" \
    "datasets>=2.20.0" "huggingface_hub>=0.25.0" "tokenizers>=0.20.0" \
    "safetensors>=0.4.0" "sentencepiece"

# 7. Constrained decoding
python3 -m pip install "outlines>=0.1.0"

# 8. API clients
python3 -m pip install "openai>=1.35.0" "google-generativeai>=0.8.0" \
    "mistralai>=1.0.0" "boto3>=1.34.0"

# 9. Data / stats
python3 -m pip install "pandas>=2.0.0" "pyarrow>=14.0.0" "numpy>=1.24.0" \
    "scipy>=1.11.0" "pyyaml>=6.0" "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" "requests>=2.31.0" "paramiko>=3.4.0"
```

All of the above is in `akash/install.sh`, which installs into the **persistent venv** at `/data/venv` and writes `INSTALL_OK` on success. On subsequent evictions, install skips if torch + flash_attn already import (~0 s).

---

## 7. Flash Attention — Compatibility and Installation

### Compatibility Matrix (what actually works)

| PyTorch | CUDA | Python | Flash Attn | ABI | Status |
|---|---|---|---|---|---|
| 2.6.0+cu124 | 12.4 | 3.10 | 2.7.4.post1 | cxx11abiFALSE | **WORKS ✅** |
| 2.5.x | 12.4 | 3.10 | 2.6.x | cxx11abiFALSE | Likely works |
| 2.6.0 | 12.4 | 3.12 | any | — | **Fails** — no cp312 wheel |
| 2.6.0 | 11.x | any | any | — | **Fails** — wheel is cu12 |

### Why `cxx11abiFALSE` matters

PyTorch wheels on Linux are compiled with the pre-C++11 ABI. Flash Attention must match. The `cxx11abiTRUE` variant is only for PyTorch compiled from source.

### Prebuilt wheel URL (fastest, most reliable)

```bash
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/\
v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python3 -m pip install "$FLASH_WHL" --no-deps
```

The `--no-deps` flag prevents pip from replacing our pinned torch 2.6.0.

### Fallback: build from source

```bash
MAX_JOBS=4 python3 -m pip install flash-attn --no-build-isolation
```

Expect ~45–60 minutes compile time on the VM.

---

## 8. Persistent Storage — Required for Production

### Why persistent `/data` is mandatory

Akash containers can be evicted for disk pressure, node load, or lease expiry. Without persistent storage:

- Installed packages are wiped (ephemeral root)
- HF model cache is lost (~50 GB re-download)
- Pipeline must restart from scratch

The production SDL mounts a **120 GiB beta3 persistent volume at `/data`**. All durable artifacts live there:

| Path | Purpose |
|---|---|
| `/data/venv` | Python virtualenv (`install.sh` is idempotent — skips if torch+flash_attn import) |
| `/data/hf_cache` | HuggingFace model cache (`HF_HOME` env var) |
| `/data/pip_cache` | pip download cache |
| `/data/state/` | Checkpoint markers (`INSTALL_OK`, `DATASET_OK`, …) |
| `/data/logs/` | install, pipeline, watchdog logs (survive eviction) |
| `/data/.env` | API keys uploaded via SFTP |

### Disk eviction vs memory OOM

Two distinct eviction triggers exist on Akash:

| Trigger | How to detect | Fix |
|---|---|---|
| **Disk eviction** | `watchdog.log` shows root/workspace disk near 100% before restart | Use small ephemeral root (30 GiB) + persistent `/data`; redirect all caches to `/data` |
| **Memory OOM** | `memory.current` near `memory.max` during model load | Set SDL `memory: 64Gi`; use smaller models if node-level eviction persists |

After any eviction, check `/data/logs/watchdog.log` — the last lines before restart show which resource was spiking.

### Fallback (ephemeral-only deployments)

If persistent storage is unavailable, chain install + predownload + pipeline in a **single tmux session** so a mid-run restart loses everything anyway but at least one session completes if the container stays alive:

```bash
tmux new-session -d -s pipeline \
  'cp /data/.env /data/Audit_Benchmark/Code/mirage/.env \
   && bash /data/Audit_Benchmark/akash/install.sh \
   && /data/venv/bin/python /data/Audit_Benchmark/akash/predownload_models.py \
   && /data/venv/bin/python /data/Audit_Benchmark/akash/_full_pipeline.py; echo DONE'
```

Do not use ephemeral-only storage for multi-hour GPU runs.

---

## 9. Memory Management — cgroup Limits vs free -h

> **This section exists because `free -h` lied to us and wasted hours of debugging.**

### The `free -h` lie — critical to understand

When you SSH into the container and run `free -h`, you will see something like:

```
               total        used        free      shared  buff/cache   available
Mem:           2.0Ti        87Gi       1.2Ti       5.1Gi       687Gi       1.9Ti
```

**THIS IS WRONG. Your container does NOT have 2 TiB of RAM.**

`free -h` reads `/proc/meminfo`, which is a global kernel interface showing the **entire host machine's RAM** — not your container's limit. It is completely blind to cgroup memory limits.

### How to check your REAL memory limit

```bash
# cgroup v2 (most Akash providers)
cat /sys/fs/cgroup/memory.max
# Output: 68719476736 = 64 GiB  ← this is your ACTUAL limit

# cgroup v2 — current usage
cat /sys/fs/cgroup/memory.current
# Output: 265740288 = ~253 MB   ← your actual usage
```

To convert bytes to GiB in Python:
```python
bytes_val = 68719476736
print(f"{bytes_val / 1024**3:.1f} GiB")  # → 64.0 GiB
```

### What happens when the limit is too low (16 GiB)

With `memory: 16Gi` in the SDL, these events unfold silently:

1. Container boots, SSH works — all looks fine
2. `free -h` shows 2 TiB — looks like plenty of RAM
3. install.sh runs — packages install fine (~5 GB used)
4. LLaMA-3.1-8B loads into GPU — also needs ~14 GB CPU RAM buffer during loading → **RAM spikes near 16 GB limit**
5. Qwen2.5-7B starts loading → **cgroup OOM killer fires**
6. Container receives SIGTERM, restarts silently
7. All packages are wiped (ephemeral storage)

**The user sees:** container restarting randomly with no error message. This is a known Akash limitation — [GitHub issue #246](https://github.com/akash-network/support/issues/246) is still open as of 2026. Akash tenants cannot see the pod-level exit code (137 = OOM); only the provider operator can see it via `kubectl describe pod`.

### Why model loading uses CPU RAM even though the model goes to GPU

HuggingFace `transformers` with `device_map="auto"` uses Big Model Inference:
1. Creates model skeleton on PyTorch meta device (~0 RAM)
2. Loads weights from disk → **allocates CPU RAM buffer** for each layer
3. Dispatches layer to GPU → frees CPU buffer

For a 7B model in bfloat16: 7B × 2 bytes = **14 GB of CPU RAM** temporarily allocated during loading, even though the final model lives in GPU VRAM.

### Required RAM by model

| Model | VRAM (bf16) | CPU RAM during loading |
|---|---|---|
| Llama-3.1-8B-Instruct | ~16 GB | ~14 GB peak |
| Qwen2.5-7B-Instruct | ~14 GB | ~12 GB peak |
| Gemma-2-2b-it | ~4 GB | ~4 GB peak |
| Phi-4-mini-instruct | ~8 GB | ~7 GB peak |

**Minimum safe container RAM:** largest model peak + OS + Python overhead = ~14 + 3 = **17 GB**. Use **64 GiB** in the SDL.

> **Rule:** Always set `memory: 64Gi`. Values below 17 GiB trigger silent cgroup OOM during model loading.

### Issue 7: Container restarts with no error during model loading

**Symptoms:**
- Container runs fine for 3–5 minutes, then restarts silently
- `free -h` shows 2 TiB available — looks like no memory issue
- Log file disappears after restart (ephemeral storage wiped)
- Models that loaded previously cannot be found after restart

**Diagnosis checklist:**
```bash
# 1. Check your REAL cgroup memory limit
cat /sys/fs/cgroup/memory.max
# If this shows < 68719476736 (64 GiB), you have too little RAM allocated in SDL

# 2. Check current memory usage
cat /sys/fs/cgroup/memory.current

# 3. Check how long container has been alive (PID 1 uptime in seconds)
ps -p 1 -o etimes=
# If this is very small (< 300s) and you expected it to be running for hours,
# the container was restarted recently

# 4. Check host load (high load = host node is overloaded, may cause node-level eviction)
cat /proc/loadavg
# If load > 8.0, the host is under heavy load from many tenants
```

**Root cause determination:**

| What you see | Root cause | Fix |
|---|---|---|
| `memory.max` < 17 GB | Container OOM (cgroup limit too low) | Increase `memory:` in SDL to `64Gi` and redeploy |
| `memory.max` = 64 GB but `memory.current` < 1 GB at restart | Node-level eviction (host OOM) | Reduce model size (see §9 below) or get a different provider |
| Load average > 8.0 | Provider node overloaded by other tenants | Wait for off-peak time or deploy to a different provider |

### Issue 8: Node-level eviction even with 64 GiB container RAM

**What happens:** Even with 64 GiB cgroup limit and only 300 MB used in the container, the host node's overall memory (including GPU kernel allocations from ALL tenants) runs low. The Kubernetes kubelet evicts our pod to reclaim resources for the node.

**Why GPU model loading causes host-level pressure:**
When a model loads onto the A100, the CUDA driver allocates **pinned memory** (locked physical pages bridging CPU and GPU memory). These allocations happen at the **kernel level** and are:
- NOT counted in your container's cgroup memory usage
- Counted against the **host node's total physical memory**
- Contributed by ALL tenants' GPU workloads on the same server

On an overloaded Akash provider node (10+ load average), multiple tenants running GPU models simultaneously can exhaust the host's pinned memory, causing the kubelet to evict pods.

**The fix — two complementary approaches:**

**Approach A: Reduce individual model size (implemented in MIRAGE)**

We replaced `google/gemma-2-9b-it` (9B params, ~18 GB VRAM) with `google/gemma-2-2b-it` (2B params, ~4 GB VRAM). Same Gemma-2 architecture, same `-it` instruction-tuned variant, same TransformerLens compatibility. Total VRAM across all 4 models dropped from ~56 GB to ~42 GB, and CPU RAM peaks dropped proportionally.

```
Before: Llama-8B(16) + Qwen-7B(14) + Gemma-9B(18) + Phi-4mini(8) = ~56 GB VRAM
After:  Llama-8B(16) + Qwen-7B(14) + Gemma-2B(4)  + Phi-4mini(8) = ~42 GB VRAM
```

**Approach B: Sequential model loading (load → test → unload → next)**

Instead of keeping all 4 models in VRAM simultaneously (cumulative ~42 GB), the dry run now loads one model, tests it, unloads it, then loads the next. Peak VRAM at any time = largest single model (~16 GB), not the sum.

```python
# dry_run_gpu_cpu.py — after each model test:
finally:
    unload_model(model_cfg["name"])   # frees VRAM + forces GPU cache clear
```

This is implemented in `Dry_Run/dry_run_gpu_cpu.py`. The production `load_all_osm_models()` still uses the keep-all strategy for throughput once models are cached.

### Issue 9: Accumulation of models in VRAM (keep-all strategy in dry run)

**Original strategy:** `load_all_osm_models()` loads all 4 models and keeps them in VRAM.

| Time | Action | VRAM used |
|---|---|---|
| t=0 | Load LLaMA-8B | 16 GB |
| t=1 | Load Qwen-7B (LLaMA still in VRAM) | 30 GB |
| t=2 | Load Gemma-9B (both still in VRAM) | 48 GB ← triggers GPU memory pressure |
| t=3 | Load Phi-mini (all still in VRAM) | 56 GB |

**Fixed strategy (dry run only):** sequential load/unload per model. Peak VRAM = 16 GB at any time.

The production pipeline uses keep-all because it needs all models available for each probe. But during dry run validation, sequential is both safer and faster.

### Quick reference — what to check when container restarts unexpectedly

```bash
# Run these immediately after SSH-ing into a restarted container:

echo "=== Container uptime ==="
ps -p 1 -o etimes=

echo "=== Real RAM limit ==="
python3 -c "
limit = int(open('/sys/fs/cgroup/memory.max').read())
used = int(open('/sys/fs/cgroup/memory.current').read())
print(f'Limit: {limit/1024**3:.1f} GiB')
print(f'Used:  {used/1024**3:.1f} GiB')
print(f'free -h would show: WRONG (it shows host RAM, not this)')
"

echo "=== Host load ==="
cat /proc/loadavg
```

---

## 10. Pre-Downloading All Models Before Running Code

> **Golden rule: ALWAYS pre-download ALL models AND datasets before any GPU code runs.**
> If any model or dataset is missing at runtime, the pipeline crashes mid-run — potentially hours into an expensive GPU session.

### Why this matters on Akash

On conventional cloud (AWS, GCP), if a download fails you lose seconds. On Akash:
- Downloads happen while the GPU is active → CUDA context is live → high host-memory pressure
- A 14 GB model download + an active CUDA context can trigger **node-level container eviction** (exit 137) on an overloaded Akash provider — silently terminating the container mid-run
- All installed packages and partial downloads are wiped (ephemeral storage)
- You re-pay for the entire pipeline restart

**The solution is to pre-download everything in a dedicated phase BEFORE any GPU model is loaded.** `predownload_models.py` does this. Pure disk I/O, no GPU activity, no CUDA context.

### Why predownload must run before GPU

On Akash, a mid-run model download while CUDA is active increases host pinned-memory pressure and can trigger container eviction. All installed packages and partial state on ephemeral root are lost. Pre-downloading to the persistent `/data/hf_cache` before Step 2/4 of the GPU pipeline eliminates this risk.

### The predownload_models.py script

```python
# akash/predownload_models.py  (simplified)
from huggingface_hub import snapshot_download

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",   # ~16 GB  (gated — needs HF token)
    "Qwen/Qwen2.5-7B-Instruct",            # ~14 GB  (public)
    "google/gemma-2-2b-it",                # ~4 GB   (public)
    "microsoft/Phi-4-mini-instruct",       # ~8 GB   (public)
]

for model_id in MODELS:
    snapshot_download(repo_id=model_id, token=HF_TOKEN)
```

`snapshot_download` is cache-aware: if a model is already fully downloaded it skips it immediately. So on a restart where LLaMA was already cached, only the missing models are fetched.

### Critical rules for model usage

| Rule | Reason |
|---|---|
| Download ALL 4 models before any GPU code | Mid-run download + CUDA context risks eviction |
| Cache on `/data/hf_cache` (persistent) | Survives container eviction; no re-download |
| Use `snapshot_download` (not single-file download) | Gets tokenizer, config, and all weight shards |
| Keep `HF_TOKEN` in `/data/.env` before predownload | LLaMA and Gemma are gated models |
| Keep `config.py` and `predownload_models.py` in sync | Mismatch causes runtime cache miss or wasted downloads |
| Never add a model to one file without updating the other | Same matched pair rule as before |

### Keeping config.py and predownload_models.py in sync

`config.py` defines what models the pipeline uses. `predownload_models.py` defines what gets pre-cached. **These two lists must always match.**

```
config.py  OSM_MODELS[*]["hf_id"]   ←→   predownload_models.py  MODELS[]
```

Whenever you change one, change the other. If you add a new model to the research, update both files together — they are a matched pair.

### Dataset pre-download (same principle)

The pipeline uses 4 benchmark datasets: `BBQ`, `CrowS-Pairs`, `StereoSet`, `WinoBias`. These are loaded via HuggingFace `datasets`. To pre-cache them:

```bash
# Run on the VM before any pipeline code (no GPU needed)
python3 -c "
from datasets import load_dataset
for ds in ['heegyu/bbq', 'BigSocialMedia/crows-pairs',
           'McGill-NLP/stereoset', 'uclanlp/winobias']:
    try:
        load_dataset(ds, split='test', trust_remote_code=True)
        print(f'OK: {ds}')
    except Exception as e:
        print(f'WARN {ds}: {e}')
"
```

Add this to `predownload_models.py` (or a companion `predownload_datasets.py`) if datasets are not already bundled in the repo.

### The correct pipeline execution order

```
1. Deploy VM with persistent /data volume (§4)
2. Upload .env to /data/.env (§11)
3. Supervisor or autonomous_guard → _full_pipeline.py:
   a. install.sh          → INSTALL_OK   (venv at /data/venv)
   b. predownload_models  → PREDOWNLOAD_OK (models at /data/hf_cache)
   c. patch_slot_b_only OR skip det patch if valid → regenerate_api_slots → DATASET_OK
   d. run_gpu_pipeline    → GPU_PIPELINE_OK + PIPELINE_COMPLETE
4. Download results; run CPU_Only/ locally (§18)
```

During step 3c, **`autonomous_guard.sh`** can run instead of the supervisor: it waits for regen, validates the pentad, then starts the supervisor. Never run the supervisor concurrently with an active `regenerate_api_slots.py` unless the guard is managing the handoff.

`predownload_models.py` writes per-model marker files in `$STATE_DIR` so each model downloads exactly once per lease. Partial downloads resume via `snapshot_download` cache semantics.

Optional gate: set `MIRAGE_RUN_DRYRUN=1` to insert a 2-seed dry run before step 4d.

---

## 11. Uploading .env to the VM

The `.env` file contains API keys and must never be committed to git.

### Upload via SFTP (implemented in `akash/_deploy_mirage.py`)

```python
import paramiko
from pathlib import Path

LOCAL_ENV = Path("Code/mirage/.env")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
               timeout=30, banner_timeout=60)
sftp = client.open_sftp()
sftp.put(str(LOCAL_ENV), "/data/.env")   # staging path on persistent volume
sftp.close()
```

The supervisor copies `/data/.env` → `Code/mirage/.env` and strips Windows CRLF (`\r\n`) from every value. **Upload `.env` before the supervisor's 600 s wait expires** so gated model downloads succeed.

Required keys in `Code/mirage/.env`:

| Key | Purpose |
|---|---|
| `HUGGINGFACE_TOKEN` / `HF_TOKEN` | Gated models (LLaMA, Gemma) |
| `DEEPSEEK_API_KEY` | Slot d/e generation (context shift + CoT attack) |
| `DEEPSEEK_API_KEY_2` | Second DeepSeek key for parallel regen workers |
| `AKASH_API_KEY` | Deployment scripts (local only) |
| `MIRAGE_GIT_PULL` | Set to `1` to enable `git pull` on boot (default: off) |

---

## 12. Production Pipeline — Markers, Supervisor, GPU Work

### Checkpoint markers (`/data/state/`)

The pipeline is driven by marker files. Each step runs only if its marker is absent. Completed steps are skipped instantly on resume.

| Marker | Step | What it means |
|---|---|---|
| `INSTALL_OK` | `akash/install.sh` | Persistent venv at `/data/venv` with torch + flash_attn |
| `PREDOWNLOAD_OK` | `akash/predownload_models.py` | All 4 OSM models in `/data/hf_cache` |
| `DATASET_OK` | Pentad build + validation | Full 12-slot `pentad_dataset.parquet` passes `assert_production_ready()` |
| `GPU_PIPELINE_OK` | `GPU_CPU/run_gpu_pipeline.py` | Behavioral eval + CDVA + tau calibration complete |
| `PIPELINE_COMPLETE` | Final sentinel | All production GPU steps succeeded |

Optional: `DRYRUN_OK` — set only when `MIRAGE_RUN_DRYRUN=1` forces a 2-seed dry run gate.

**Rules:**

- Never set `DATASET_OK` manually — `_full_pipeline.py` validates before writing it.
- If pentad fails validation after `DATASET_OK` exists, the orchestrator **clears** `DATASET_OK` and `GPU_PIPELINE_OK` automatically.
- `CPU_Only/` scoring runs **locally after download** — not on the VM.

### Orchestrator flow (`akash/_full_pipeline.py`)

```
INSTALL_OK        → install.sh (idempotent venv)
PREDOWNLOAD_OK    → predownload_models.py (disk only, no GPU)
DATASET_OK        → det patch (if needed) + regenerate_api_slots.py + validation
GPU_PIPELINE_OK   → run_gpu_pipeline.py (behavioral + CDVA + tau)
PIPELINE_COMPLETE → final marker
```

`_ensure_dataset()` enforces:

1. **`_det_slots_valid()`** — if deterministic slots a/b/c already pass validation (including `validate_slot_b_grammar()`), **skip** `patch_det_slots.py`
2. If slot-b only needs fixing → run `patch_slot_b_only.py` (preserves d/e; no API calls)
3. If det slots broken → run `patch_det_slots.py` (rebuilds a/b/c; **drops d/e** until regen completes)
4. Run or wait for `regenerate_api_slots.py` for DeepSeek slots d/e (timeout **43200 s** / 12 h)
5. Call `assert_production_ready()` — hard gate before GPU
6. Write `pentad_manifest.json` with SHA-256
7. Only then write `DATASET_OK`

### Slot-b vs full det patch — which script to use

| Situation | Script | d/e preserved? |
|---|---|---|
| Slot-b grammar / iso-control only | `patch_slot_b_only.py` | Yes |
| Equivalence-set or slot-c logic changed | `patch_det_slots.py` | No — must regen d/e |
| d/e missing but det valid | *(skip patch)* + `regenerate_api_slots.py --keep-checkpoint` | N/A |

**Rule:** Never run `patch_det_slots.py` when det slots already validate — it saves det-only rows and removes all d/e until regen finishes.

### Autonomous guard (`akash/autonomous_guard.sh`)

Runs on the VM during dataset rebuild (60 s poll loop):

- Waits while `regenerate_api_slots.py` is active
- Restarts dead regen with `--keep-checkpoint`
- On `assert_production_ready()` pass → starts `supervise_pipeline.sh`
- Keeps supervisor off until the pentad reaches 7,152 rows

```bash
nohup bash /data/Audit_Benchmark/akash/autonomous_guard.sh \
  >> /data/logs/autonomous_guard.log 2>&1 &
```

### Git pull protection (`supervise_pipeline.sh`)

`git pull` on container boot is **disabled by default** (`MIRAGE_GIT_PULL=0`). Uploaded hotfixes to `/data/Audit_Benchmark` survive restarts. Set `MIRAGE_GIT_PULL=1` in the SDL env only when you intentionally want to sync from GitHub.

### Deploy from local machine

```bash
# From repo root:
python akash/_deploy_mirage.py    # creates deployment, uploads .env, prints SSH
python akash/_monitor.py          # polls /data/logs until PIPELINE_COMPLETE
```

### Monitor on the VM

```bash
ssh root@<provider-host> -p <port>
# Password: MirageVM2026!

tail -f /data/logs/pipeline_attempt_1.log   # main pipeline log
tail -f /data/logs/watchdog.log             # resource snapshots every 10 s
tail -f /data/logs/supervise.log              # supervisor retry loop
ls -la /data/state/                           # checkpoint markers
```

### GPU pipeline steps (`GPU_CPU/run_gpu_pipeline.py`)

```
Step 1/4: Load all 4 OSM models (~42 GB VRAM on A100 80GB)
Step 2/4: Behavioral evaluation — 4 models × 7152 prompts (det + variance)
Step 3/4: CDVA patching
Step 4/4: Tau calibration
```

`behavioral_results.parquet` is written after each model completes its full deterministic pass (not incrementally per batch). Absence of result files during Step 2 is normal.

### Expected runtime

| Phase | Duration |
|---|---|
| Install (first time) | ~2–3 min |
| Pre-download (~42 GB) | ~5–10 min |
| Dataset build (DeepSeek d/e for 596 seeds) | ~1–2 hr |
| GPU behavioral (4 models × 7152 + variance) | ~4–5 hr |
| CDVA + tau calibration | ~1–2 hr |
| **Total GPU pipeline** | **~6–10 hr** |

At ~147 prompts/min observed on A100, budget **~6 hours** for remaining GPU work after dataset is ready.

### Optional dry run

Set `MIRAGE_RUN_DRYRUN=1` in the SDL env to force a 2-seed dry run gate before GPU work. Production deployments skip dry run (validated separately).

---

## 13. Dataset Validation Gates (Pentad Integrity)

The pentad dataset (`Dataset/seeds/pentad_dataset.parquet`) is the foundation of all MIRAGE results. **Never start GPU inference on an unvalidated pentad.**

### Production-ready requirements

`Dataset/validate_pentad.py` → `assert_production_ready()` enforces:

| Check | Requirement |
|---|---|
| Audit sources only | BBQ, CrowS-Pairs, StereoSet in main set |
| WinoBias held out | Zero WinoBias rows in pentad (separate file) |
| Row count | `n_seeds × 12` audit rows (currently **596 × 12 = 7,152**) |
| Slot completeness | Each seed: a(1) + b(1) + c(5) + d(2) + e(3) = 12 |
| Prompt text | No `"None"`, `"nan"`, `"null"` sentinel strings |
| Slot-b differs from slot-a | Protected-token swap must change the text |
| Slot-b grammar | No `Person and Person`, `person man`, `Context: person`, `person are`, etc. |
| Slot-c variants distinct | All 5 counterfactual swaps must differ |
| MCQ options | BBQ prompts include `(A)`, `(B)`, `(C)` options |
| Gold answers | Scorable per `Dataset/gold_utils.py` (BBQ `"Unknown"` is valid) |
| DeepSeek slots d/e | Must embed slot-a text; validated by `validate_deepseek_embeds_slot_a()` |

### Seed counts (actual production build)

| Source | Included seeds |
|---|---|
| BBQ | 254 |
| CrowS-Pairs | 181 |
| StereoSet | 161 |
| **Total audit (N)** | **596** |
| WinoBias | 200 (held out — `winobias_seeds.parquet`) |
| Excluded StereoSet | 22 (documented in `excluded_seeds.json`) |

Report **N = 596** in the paper. Check `Dataset/seeds/excluded_seeds.json` for exclusion reasons.

### Slot-b iso-control neutralization (`pentad_generator.py`)

Slot-b replaces protected demographic tokens with neutral terms. Production fixes ensure grammatical iso-controls:

| Source pattern | Fix |
|---|---|
| BBQ `"Person and Person"` | Distinct `Person A/B/C` via `_neutralize_bbq_names()` |
| CrowS `"person man"` | `_expand_crows_surface()` + `_replace_with_neutral_person()` |
| StereoSet `"The person man"` (e.g. Muslim man) | Expand phrase; replace full compound → `person` |
| `"Context: person is..."` | `_polish_slot_b_text()` → `Context: A person is...` |
| `"Gentlemen are"` → `"person are"` | `_PLURAL_NEUTRAL_TERMS` → `People are` |

Validate after any slot-b change:

```bash
python -c "
import pandas as pd
from Dataset.validate_pentad import validate_slot_b_grammar
validate_slot_b_grammar(pd.read_parquet('Dataset/seeds/pentad_dataset.parquet'))
print('slot-b grammar OK')
"
```

To patch slot-b without touching d/e:

```bash
python patch_slot_b_only.py
# Only regen d/e if slot-a text changed:
python regenerate_api_slots.py --keep-checkpoint
```

### Slot distribution (596 seeds)

```
a=596, b=596, c=2980 (596×5), d=1192 (596×2), e=1788 (596×3)
```

### Manifest and SHA tracking

After validation, `write_pentad_manifest()` writes `Dataset/seeds/pentad_manifest.json`:

```json
{
  "pentad_sha256": "f8ddb67d6032...",
  "n_rows": 7152,
  "n_audit_seeds": 596,
  "rows_per_seed": 12,
  "has_api_slots": true,
  "excluded_seeds": { ... }
}
```

The SHA is used by `pipeline_guards.py` to detect stale GPU results (see §15).

### Manual validation (on VM or locally)

```bash
cd Code/mirage
python -c "
import pandas as pd
from Dataset.validate_pentad import assert_production_ready
assert_production_ready(pd.read_parquet('Dataset/seeds/pentad_dataset.parquet'))
print('PRODUCTION READY')
"
```

### Scoring gold rules (`Dataset/gold_utils.py`)

- Empty, `"nan"`, `"none"` → not scorable
- BBQ labels like `"Unknown"` and `"Can't be determined"` → **valid gold** (ambiguous items)
- Non-BBQ sources with `"unknown"` → not scorable (construction failure)
- `CPU_Only/scoring.py` uses `is_scorable_gold()` — never auto-pass on missing gold

---

## 14. DeepSeek API Slots (d/e) Regeneration

Slots **d** (context shift) and **e** (CoT attack) are generated by DeepSeek API calls. They depend on correct slot-a text.

### Parallel workers and dual keys

`context_shift_drafter.py` and `cot_attack_generator.py` use **2 parallel workers**, one per DeepSeek key (`DEEPSEEK_API_KEY`, `DEEPSEEK_API_KEY_2`). Each seed falls back to the alternate key on failure. Retries: 5 per seed.

Observed throughput: ~50–60 seeds/min for slot-d on Akash.

### Build order

```
1. sample_seeds.py         → seeds.parquet (BBQ + CrowS + StereoSet; WinoBias separate)
2. pentad_generator.py     → initial pentad
   OR patch_slot_b_only.py → slot-b only (preserves d/e)
   OR patch_det_slots.py    → a/b/c only (drops d/e — use only when det broken)
3. regenerate_api_slots.py → DeepSeek slots d/e for all audit seeds
4. validate_pentad.py       → assert_production_ready() + validate_slot_b_grammar()
5. write_pentad_manifest    → SHA + metadata
```

### Running regeneration

```bash
cd /data/Audit_Benchmark/Code/mirage
/data/venv/bin/python regenerate_api_slots.py
```

Flags:

| Flag | Effect |
|---|---|
| (default) | Clears stale checkpoints; regenerates all d/e |
| `--keep-checkpoint` | Resume from JSON checkpoint — safe when slot-a text unchanged |
| `--dry-run` | Validate inputs without API calls |

Checkpoints live at `Dataset/seeds/context_shift_checkpoint.json` and `cot_attack_checkpoint.json`.

**Incremental save:** `regenerate_api_slots.py` writes the pentad after slot-d completes (`_save_partial_pentad()`), so a crash during slot-e does not lose slot-d progress. Checkpoints are kept until full validation passes.

### Orchestrator integration

`_full_pipeline.py` → `_ensure_dataset()`:

- Skips `patch_det_slots.py` when `_det_slots_valid()` passes (includes slot-b grammar)
- Runs `patch_slot_b_only.py` when orchestrator detects slot-b-only fixes needed
- Starts or waits for `regenerate_api_slots.py` if d/e missing
- Polls every 30 s (timeout **43200 s** / 12 h) until `assert_production_ready()` passes
- Never proceeds to GPU on a det-only partial set (e.g. 4,172 rows without d/e)

**Before any pentad patch on a running VM:**

```bash
pkill -f supervise_pipeline
pkill -f _full_pipeline
```

---

## 15. GPU Pipeline Guards & Stale Result Prevention

`GPU_CPU/pipeline_guards.py` → `clear_stale_gpu_results_if_pentad_changed()` runs at GPU pipeline start.

**Logic:**

1. Compute SHA-256 of current `pentad_dataset.parquet`
2. Compare to `pentad_manifest.json` → `pentad_sha256`
3. If SHA changed (or manifest missing):
   - Delete `results/behavioral_results.parquet`
   - Delete `results/cdva_results.parquet`
   - Delete `results/tau_calibration.json`
   - Clear `GPU_PIPELINE_OK` marker

This prevents scoring behavioral outputs from a prior dataset version.

`run_gpu_pipeline.py` also calls `assert_production_ready()` before loading any model — a second hard gate at GPU entry.

---

## 16. Reset Protocol — Resume from Last Good Stage

When the pipeline stops or data may be corrupt, reset **only from the last verified-good stage**. Never blindly delete all markers.

### Decision tree

```
1. Run assert_production_ready() on pentad
   ├─ PASSES → keep DATASET_OK; only clear GPU markers if results stale
   └─ FAILS  → clear DATASET_OK + GPU_PIPELINE_OK; rebuild dataset

2. Check pentad_manifest.json SHA vs current pentad SHA
   ├─ MATCH   → GPU results may be valid; resume from GPU_PIPELINE_OK if present
   └─ DIFFER  → pipeline_guards clears results + GPU_PIPELINE_OK automatically

3. Restart supervisor (only after pentad validates)
   bash /data/Audit_Benchmark/akash/supervise_pipeline.sh
   # Or let autonomous_guard.sh start it automatically
```

### Manual reset commands (use sparingly)

```bash
# Pentad still validates — restart GPU only:
pkill -f supervise_pipeline
rm -f /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
bash /data/Audit_Benchmark/akash/supervise_pipeline.sh

# Slot-b fix only (det + d/e intact):
pkill -f supervise_pipeline
cd /data/Audit_Benchmark/Code/mirage
/data/venv/bin/python patch_slot_b_only.py
# Regen d/e only if slot-a text changed:
/data/venv/bin/python regenerate_api_slots.py --keep-checkpoint

# Det slots broken — full det rebuild + regen:
pkill -f supervise_pipeline
rm -f /data/state/DATASET_OK /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
cd /data/Audit_Benchmark/Code/mirage
/data/venv/bin/python patch_det_slots.py
/data/venv/bin/python regenerate_api_slots.py --keep-checkpoint
nohup bash /data/Audit_Benchmark/akash/autonomous_guard.sh \
  >> /data/logs/autonomous_guard.log 2>&1 &

# Never do this on a valid pentad:
# python run_dataset.py --force   ← bypasses validation gates
```

### What markers to keep vs clear

| Situation | Keep | Clear |
|---|---|---|
| GPU interrupted mid-run, pentad valid | `INSTALL_OK`, `PREDOWNLOAD_OK`, `DATASET_OK` | `GPU_PIPELINE_OK`, `PIPELINE_COMPLETE` |
| Slot-b patched, d/e unchanged | `INSTALL_OK`, `PREDOWNLOAD_OK`, `DATASET_OK` | `GPU_PIPELINE_OK`, `PIPELINE_COMPLETE` (if SHA changed) |
| Pentad regenerated (d/e rebuilt) | `INSTALL_OK`, `PREDOWNLOAD_OK` | `DATASET_OK`, `GPU_PIPELINE_OK`, `PIPELINE_COMPLETE` |
| Fresh deploy on new VM | (none — all rebuilt) | — |

---

## 17. Monitoring & Health Checks

### Local monitoring scripts (run from repo root)

| Script | Purpose |
|---|---|
| `python akash/_monitor.py` | Poll until `PIPELINE_COMPLETE` |
| `python akash/_pipeline_health.py` | Full audit: markers, pentad, GPU progress, ETA |
| `python akash/_vm_progress.py` | Quick marker + pentad + log snapshot |
| `python akash/_regen_progress.py` | DeepSeek checkpoint progress (slot-d/e counts) |
| `python akash/_quick_eta.py` | Progress rate and hours remaining |
| `python akash/_monitor_regen.py` | Watch DeepSeek slot regeneration |
| `python akash/_deploy_hardened.py` | Upload fixes, restart regen, start autonomous guard |
| `python akash/_deep_audit.py` | Research validity audit (prompts, gold, scoring) |
| `python akash/_research_audit.py` | Dataset + metrics research audit |

### On-VM log locations

| Log | Content |
|---|---|
| `/data/logs/pipeline_attempt_N.log` | Full pipeline output (one file per supervisor attempt) |
| `/data/logs/watchdog.log` | Memory, disk, GPU every 10 s — check before eviction |
| `/data/logs/install.log` | Package install output |
| `/data/logs/supervise.log` | Supervisor retry loop |
| `/data/logs/autonomous_guard.log` | Autonomous guard poll loop |
| `/data/Audit_Benchmark/LOG/regen_api_slots.log` | DeepSeek regeneration detail |

### Health checklist

```bash
# Markers
ls -la /data/state/

# Pentad valid?
cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python -c \
  "import pandas as pd; from Dataset.validate_pentad import assert_production_ready; \
   assert_production_ready(pd.read_parquet('Dataset/seeds/pentad_dataset.parquet')); print('OK')"

# GPU running?
pgrep -af run_gpu_pipeline

# Latest progress
grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -3

# GPU utilisation
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

---

## 18. Post-GPU: CPU-Only Scoring (Local)

After `PIPELINE_COMPLETE` on the VM:

1. Download results:
   ```bash
   scp -P <port> root@<host>:/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet .
   scp -P <port> root@<host>:/data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet .
   scp -P <port> root@<host>:/data/Audit_Benchmark/Code/mirage/results/tau_calibration.json .
   ```
2. Download validated pentad + manifest (for reproducibility):
   ```bash
   scp -P <port> root@<host>:/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet .
   scp -P <port> root@<host>:/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_manifest.json .
   ```
3. Run CPU scoring and analysis locally (from `Code/mirage/`):
   ```bash
   # Scoring, statistics, leaderboard, predictive validity — run each module
   # after placing downloaded parquets in results/
   python -m CPU_Only.scoring
   python -m CPU_Only.results_analysis
   python -m CPU_Only.statistics
   python -m CPU_Only.leaderboard
   python -m CPU_Only.predictive_validity
   ```
   Or use `Dry_Run/dry_run_cpu_only.py` for a single-seed sanity check first.

4. Close the Akash deployment to stop billing:
   ```python
   import requests
   requests.delete(f"https://console-api.akash.network/v1/deployments/{dseq}",
                   headers={"x-api-key": AKASH_API_KEY})
   ```

---

## 19. Scripts Reference

All scripts live in `akash/`. Run from repo root.

| Script | Purpose |
|---|---|
| `_deploy_mirage.py` | **Primary deploy.** SDL → bids → lease → SFTP `.env` → print SSH |
| `_full_pipeline.py` | On-VM orchestrator (called by supervisor, not run locally) |
| `supervise_pipeline.sh` | Pipeline supervisor — retries on crash, skips completed markers |
| `autonomous_guard.sh` | On-VM guard: wait for regen, validate, start supervisor |
| `watchdog.sh` | Resource logger to `/data/logs/watchdog.log` every 10 s |
| `install.sh` | Idempotent venv install to `/data/venv` |
| `predownload_models.py` | Pre-download all 4 OSM models to `/data/hf_cache` |
| `_monitor.py` | Poll logs until `PIPELINE_COMPLETE` |
| `_pipeline_health.py` | Full health audit + ETA (markers, pentad, GPU, logs) |
| `_vm_progress.py` | Quick progress snapshot |
| `_quick_eta.py` | Progress rate and hours remaining |
| `_monitor_regen.py` | Watch DeepSeek slot-d/e regeneration |
| `_upload_mirage_fixes.py` | Upload local code fixes to running VM |
| `_deploy_hardened.py` | Deploy hardened pipeline + start autonomous guard |
| `_slotb_fix_restart.py` | Stop supervisor, patch slot-b, regen, validate, restart |
| `_prelaunch_audit_clean.py` | Upload code, patch, start regen, validate, clean, launch |
| `_prelaunch_finish.py` | Poll regen → validate → launch (12 h timeout) |
| `_repair_and_restart.py` | Stop GPU, clear bad markers, rebuild dataset, restart |
| `_diagnose2.py` | Deep diagnostics: PID 1 uptime, cgroup memory, disk, host load |
| `_quick_check.py` | Fast check: VM reachable? processes? markers? |
| `vm_ssh.txt` | VM host/port/dseq (gitignored, updated per deployment) |

### VM connection template

| Field | Value |
|---|---|
| SSH | `ssh root@<provider-host> -p <port>` |
| Password | `MirageVM2026!` |
| RAM | 64 GiB (verify: `cat /sys/fs/cgroup/memory.max`) |
| Repo path | `/data/Audit_Benchmark` |
| State markers | `/data/state/` |
| Results | `/data/Audit_Benchmark/Code/mirage/results/` |

---

## 20. Troubleshooting Index

### Dataset / pentad issues

| Symptom | Cause | Fix |
|---|---|---|
| GPU starts on partial pentad (no d/e rows) | `DATASET_OK` set without validation or supervisor ran during regen | Kill supervisor; run `regenerate_api_slots.py --keep-checkpoint`; use `autonomous_guard.sh`; verify with `assert_production_ready()` |
| Pentad dropped to 4,172 rows | `patch_det_slots.py` ran while d/e existed | Never run `patch_det_slots` when det valid; regen d/e with `--keep-checkpoint` |
| Slot-b ungrammatical (`person man`, etc.) | Incomplete neutralization | Run `patch_slot_b_only.py`; `validate_slot_b_grammar()` must pass |
| Slot-b identical to slot-a | Broken protected-token swap | Re-run `patch_slot_b_only.py` or `patch_det_slots.py` if equivalence sets changed |
| Uploaded fixes overwritten on restart | `git pull` on boot | Keep `MIRAGE_GIT_PULL=0` (default); upload via `_deploy_hardened.py` |
| Scoring compares against empty gold | Missing or placeholder `gold_answer` | Rebuild pentad; use `gold_utils.is_scorable_gold()` — BBQ `"Unknown"` is valid |
| WinoBias rows in main pentad | Seeds not filtered at build | WinoBias must be in separate file only; `assert_production_ready()` rejects WinoBias rows |
| Slot-c all identical | Degenerate counterfactual swaps | Re-run `pentad_generator.py` (source-aware CrowS diff pairing) |
| DeepSeek d/e don't embed slot-a | Stale API checkpoints | Re-run `regenerate_api_slots.py` (default clears checkpoints) |
| `"None"` in StereoSet prompts | Unresolved template placeholder | `validate_pentad.py` rejects sentinel strings; rebuild affected seeds |
| GPU results don't match current pentad | Pentad SHA changed after GPU run | `pipeline_guards.py` auto-clears stale parquets + `GPU_PIPELINE_OK` |

### Pipeline marker issues

| Symptom | Cause | Fix |
|---|---|---|
| Pipeline skips dataset rebuild but pentad invalid | Stale `DATASET_OK` marker | `_full_pipeline.py` auto-clears marker when validation fails |
| Pipeline skips GPU but results are from old pentad | Stale `GPU_PIPELINE_OK` + SHA mismatch | `pipeline_guards.py` clears on SHA change; or manual reset (see §16) |
| Supervisor loops forever on dataset step | DeepSeek regen still running or failed | Check `regen_api_slots.log`; use `autonomous_guard.sh`; timeout is 12 h |
| `behavioral_results.parquet` missing mid-run | Normal — saved per model completion | Wait for model 1 deterministic pass to finish (7152 prompts) |

### Memory / container restart issues

| Symptom | First thing to check | Root cause | Fix |
|---|---|---|---|
| Container restarts every few minutes during model loading | `cat /sys/fs/cgroup/memory.max` | SDL `memory:` too low (e.g. 16Gi) — cgroup OOM | Redeploy with `memory: 64Gi` |
| `free -h` shows 2 TiB but container still OOMs | Don't trust `free -h` in containers | `free` reads host RAM, not cgroup limit | Check `/sys/fs/cgroup/memory.max` instead |
| Container restarts even with 64 GiB cgroup, only 300 MB used | `cat /proc/loadavg` | Node-level eviction (host overloaded) | Reduce model sizes; switch to sequential load/unload |
| Container evicted mid-run during a model download | Check timing: restart happens when `Fetching N files` appears | Download + active CUDA context = host memory pressure | Run `predownload_models.py` BEFORE any GPU code (see §10) |
| Log file disappears after restart | Ephemeral root wiped | Use persistent `/data` volume; logs at `/data/logs/` survive |
| `memory.max` shows correct limit but OOM still happens | Check GPU kernel memory | Pinned GPU memory not tracked by cgroup | Reduce peak VRAM by using smaller models |

### Model and dataset issues

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: model not found` or HF download mid-run | Model not pre-cached | Run `predownload_models.py` before dry run (see §10) |
| Container evicted right when a model starts downloading | Download + active CUDA context = host OOM | Pre-download all models before first GPU load; see §10 |
| `config.py` has a model not in `predownload_models.py` | Lists out of sync | Keep `config.py OSM_MODELS[*]["hf_id"]` and `predownload_models.py MODELS[]` identical |
| `predownload_models.py` downloads a model not in `config.py` | Lists out of sync | Remove the stale entry from `predownload_models.py` |
| LLaMA loads fast but Qwen/Gemma restart the container | LLaMA was cached; others were not | Pre-download ALL models, not just the ones you've tested |
| Dataset not found at runtime | HF datasets not pre-cached | Add dataset pre-download to `predownload_models.py` or a companion script (see §10) |
| `KeyError: model_name` in pipeline | Model name in code doesn't match key in `_LOADED_MODELS` | Ensure model names in `config.py`, `load_osm.py`, and `predownload_models.py` all match exactly |

### Deployment issues

| Symptom | Cause | Fix |
|---|---|---|
| No bids after 5 min | Price too low / GPU over-constrained | Raise `amount` to 10 000 000; remove `ram:`, `host:`, `signedBy:` |
| `{"code":"invalid_type","path":["manifest"]}` | Passing SDL text as manifest | Extract `data.manifest` from POST /v1/deployments response |
| SSH connection refused | Container still booting | Wait 2–3 min after lease accepted |

### Package installation issues

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pip` | Ubuntu apt pip upgraded in-place | Bootstrap from `get-pip.py` instead (see §6) |
| `import torch` fails after install | pip installed to different Python | Always use `python3 -m pip` |
| `flash_attn` compile fails | Wrong ABI, torch, or CUDA version | Use prebuilt wheel for cp310+cu124+torch2.6+cxx11abiFALSE |
| `transformer_lens.__version__` AttributeError | TL 2.18.0 has no `__version__` | Use `getattr(tl, '__version__', 'installed')` |
| `set -e` aborts install mid-way | Verification command exits non-zero | Check ALL verification lines; use `getattr` for `__version__` checks |

### Runtime issues

| Symptom | Cause | Fix |
|---|---|---|
| `HUGGINGFACE_TOKEN` missing | `.env` not on VM | Upload via SFTP BEFORE launching install chain; see §11 |
| Packages gone after reconnect | Container restarted on ephemeral root | With persistent `/data`, venv and markers survive; re-run supervisor only |
| `ModuleNotFoundError: dotenv` | Install exited early | Apply `__version__` fix to install.sh |
| OOM during CDVA patching | TL + HF model both in VRAM | Set threshold to 1.0× in `cdva_patching.py` |

### SSH / connectivity issues

| Symptom | Cause | Fix |
|---|---|---|
| `SSHException: SSH session not active` | Paramiko session timed out | Reconnect before each poll iteration |
| `UnicodeEncodeError` printing VM output | Windows cp1252 + pip progress bars | Wrap print() with `.encode("ascii", errors="replace")` |

---

## 21. Cost Estimate

| Phase | Duration | Cost at ~$1.25/hr |
|---|---|---|
| Install packages (first time) | ~2–3 min | ~$0.05 |
| Pre-download all 4 models (~42 GB) | ~5–10 min | ~$0.10–$0.20 |
| Dataset build (DeepSeek d/e, 596 seeds) | ~1–2 hr | ~$1.25–$2.50 |
| GPU behavioral eval (4 models, 7152 prompts) | ~4–5 hr | ~$5–$6.25 |
| CDVA patching + tau calibration | ~1–2 hr | ~$1.25–$2.50 |
| CPU-only scoring (local machine) | — | — |
| **Total GPU on Akash** | **~6–10 hr** | **~$8–$13** |

Close the deployment when `PIPELINE_COMPLETE` is set and results are downloaded.

---

## Appendix A: Version Pins That Are Known to Work

```
Python:           3.10.12  (system, Ubuntu 22.04 in nvidia/cuda:12.4.1 image)
CUDA:             12.4.1
torch:            2.6.0+cu124
flash_attn:       2.7.4.post1  (prebuilt wheel, cxx11abiFALSE)
transformer_lens: 2.18.0
nnsight:          0.7.0
transformers:     4.57.6
accelerate:       1.13.0
outlines:         (latest)
pandas:           2.0.3
numpy:            1.26.4
python-dotenv:    1.2.2
```

## Appendix B: OSM Model Stack (confirmed working on A100-SXM4-80GB)

| Slot | Model | Params | VRAM (bf16) | Patching lib | Status |
|---|---|---|---|---|---|
| OSM-1 | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16 GB | TransformerLens | ✅ Confirmed |
| OSM-2 | `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14 GB | nnsight | ✅ Confirmed |
| OSM-3 | `google/gemma-2-2b-it` | 2B | ~4 GB | TransformerLens | ✅ Replaced from 9B |
| OSM-4 | `microsoft/Phi-4-mini-instruct` | 3.8B | ~8 GB | nnsight | ✅ Confirmed |

**Gemma-2-2B vs Gemma-2-9B:** Use Gemma-2-2B on Akash. Same TransformerLens hooks and `-it` instruction format, but ~4 GB VRAM instead of ~18 GB — reduces node-level eviction risk during loading.

**Total VRAM across all 4 models:** ~42 GB. Fits comfortably on one A100-80GB.

## Appendix C: Pre-GPU Checklist

Run this checklist before allowing GPU work. All items must pass.

```bash
# 1. Markers
ls /data/state/INSTALL_OK /data/state/PREDOWNLOAD_OK /data/state/DATASET_OK

# 2. Pentad row count (expect 7152 = 596 seeds × 12)
/data/venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
audit = df[df.seed_source.str.lower().isin(['bbq','crows_pairs','stereoset'])]
print('rows', len(audit), 'seeds', audit.seed_id.nunique())
print('slots', audit.slot.value_counts().to_dict())
"

# 3. Production-ready gate (includes slot-b grammar)
cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python -c \
  "import pandas as pd; from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar; \
   df = pd.read_parquet('Dataset/seeds/pentad_dataset.parquet'); \
   validate_slot_b_grammar(df); assert_production_ready(df); print('OK')"

# 4. Manifest SHA present
cat /data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_manifest.json

# 5. No stale API checkpoints (should be absent after regen)
ls /data/Audit_Benchmark/Code/mirage/Dataset/seeds/*checkpoint* 2>/dev/null || echo "no checkpoints — good"

# 6. Models cached
ls /data/hf_cache/hub/models--meta-llama--Llama-3.1-8B-Instruct 2>/dev/null && echo "LLaMA cached"
```

**Hard rules before GPU:**

- Never set `DATASET_OK` manually
- Never run `run_dataset.py --force` to bypass validation
- Never start GPU on a det-only pentad (missing d/e rows)
- Never run `patch_det_slots.py` when det slots already validate
- Never run the supervisor while `regenerate_api_slots.py` is active (use `autonomous_guard.sh`)
- Never reuse DeepSeek checkpoints after slot-a text changed
- WinoBias must not appear in the main pentad
- Report **N = 596** audit seeds in the paper (22 StereoSet seeds excluded — see `excluded_seeds.json`)
