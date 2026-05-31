# Akash GPU VM Setup — MIRAGE Project
### Complete Field Guide: Deployment, Package Installation, Flash Attention, and Running Code

*Written after working through every failure mode end-to-end on live deployments.
Every "Issue" section describes something that actually broke, was debugged, and was fixed.*

---

## Table of Contents
1. [What Akash Is](#1-what-akash-is)
2. [Prerequisites](#2-prerequisites)
3. [Creating a Deployment via the Console API](#3-creating-a-deployment-via-the-console-api)
4. [SDL Configuration — What Works](#4-sdl-configuration--what-works)
5. [Container Startup Sequence](#5-container-startup-sequence)
6. [Installing Python Packages — The Right Way](#6-installing-python-packages--the-right-way)
7. [Flash Attention — Compatibility and Installation](#7-flash-attention--compatibility-and-installation)
8. [The Ephemeral Storage Problem](#8-the-ephemeral-storage-problem)
9. [Memory Management — The Most Confusing Part](#9-memory-management--the-most-confusing-part)
10. [Uploading .env to the VM](#10-uploading-env-to-the-vm)
11. [Running the Dry Run and GPU Pipeline](#11-running-the-dry-run-and-gpu-pipeline)
12. [Scripts Reference](#12-scripts-reference)
13. [Troubleshooting Index](#13-troubleshooting-index)
14. [Cost Estimate](#14-cost-estimate)

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

## 4. SDL Configuration — What Works

The deployment that successfully got an **A100-SXM4-80GB** used these exact resources:

```yaml
---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
    command:
      - bash
      - -c
      - |
        apt-get update -qq && apt-get install -y git curl tmux openssh-server wget > /dev/null
        mkdir -p /workspace
        echo "root:MirageVM2026!" | chpasswd
        mkdir -p /run/sshd
        sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
        sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
        /usr/sbin/sshd
        cd /workspace
        git clone $GITHUB_REPO Audit_Benchmark || git -C Audit_Benchmark pull
        echo "VM_READY" > /workspace/vm_ready.txt
        tail -f /dev/null
    expose:
      - port: 22
        as: 22
        to:
          - global: true

profiles:
  compute:
    mirage:
      resources:
        cpu:
          units: 4
        memory:
          size: 64Gi          # <-- CRITICAL: must be 64Gi, NOT 16Gi. See §9.
        storage:
          - size: 200Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:

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
- `memory: 64Gi` — **mandatory for GPU model workloads**. See §9 for the full explanation of why 16Gi silently kills containers.
- `cpu: 4` — sufficient; higher values reduce the number of providers that bid.
- `storage: 200Gi` — enough for packages (~5 GB) + model downloads (~50 GB).
- `gpu: units: 1` with just `vendor: nvidia:` (no model or RAM filter) — gives the most bids.

---

## 5. Container Startup Sequence

After lease is accepted, the container starts and runs the startup command:

1. `apt-get install` — basic tools (git, curl, tmux, openssh-server, wget)
2. Set root password
3. Configure and start SSH daemon (`/usr/sbin/sshd`)
4. `git clone` or `git pull` the repo
5. Write `vm_ready.txt`
6. `tail -f /dev/null` — keeps PID 1 alive so container does not exit

**SSH is available ~60–90 seconds after lease creation.**

**Important:** The startup command does NOT run `install.sh`. Python packages must be installed separately after SSH connects (see §6 and §8).

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

All of the above is in `akash/install.sh`.

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

## 8. The Ephemeral Storage Problem

**This is the single most frustrating issue when using Akash containers.**

### What happens

Akash containers have **ephemeral storage only** by default. The filesystem is wiped on every container restart. The container's startup command re-clones the repo from GitHub, but installed Python packages are gone.

### The solution: chain everything in one tmux session

```bash
# WRONG: two separate steps — restart can happen between them
# Step 1: bash install.sh
# --- container may restart here, wiping all packages ---
# Step 2: python3 dry_run.py   # ImportError: packages are GONE

# CORRECT: one chained command in a single tmux session
tmux new-session -d -s pipeline \
  'cp /workspace/mirage.env /workspace/Audit_Benchmark/Code/mirage/.env \
   && bash /workspace/Audit_Benchmark/akash/install.sh \
   && git -C /workspace/Audit_Benchmark pull --ff-only origin main \
   && cd /workspace/Audit_Benchmark/Code/mirage \
   && python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2; echo DONE'
```

This is what `akash/_full_pipeline.py` does. The `.env` copy happens first so the HF token is available for model downloads during install.

### Persistent storage (production fix)

```yaml
storage:
  - size: 200Gi       # ephemeral (root filesystem)
  - size: 100Gi       # persistent
    name: data
    mount: /data
    class: beta3       # beta3 = persistent SSD on Akash
```

Note: persistent storage costs extra and reduces the number of providers that will bid.

---

## 9. Memory Management — The Most Confusing Part

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

**Minimum safe container RAM:** largest model peak + OS + Python overhead = ~14 + 3 = **17 GB**. Use **64 GiB** to have a comfortable margin and never trigger OOM.

> **First deployment used `memory: 16Gi` → OOM on every Qwen/Gemma load.
> Changed to `memory: 64Gi` → stable. Always use 64Gi.**

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

## 10. Uploading .env to the VM

The `.env` file contains API keys and must never be committed to git.

### Upload via SFTP (implemented in `akash/_full_pipeline.py`)

```python
import paramiko
from pathlib import Path

LOCAL_ENV = Path("Code/mirage/.env")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
               timeout=30, banner_timeout=60)
sftp = client.open_sftp()
sftp.put(str(LOCAL_ENV), "/workspace/mirage.env")   # staging path outside git repo
sftp.close()
```

Then in the tmux chain, copy to runtime path:
```bash
cp /workspace/mirage.env /workspace/Audit_Benchmark/Code/mirage/.env
```

**Upload .env BEFORE running install.sh** so that HuggingFace token is available for model downloads during installation.

---

## 11. Running the Dry Run and GPU Pipeline

### Full pipeline (from local machine)

```bash
# From repo root:
python akash/_full_pipeline.py
```

This script:
1. Uploads `.env` to `/workspace/mirage.env` via SFTP
2. Kills any stale tmux sessions
3. Launches chained tmux command:
   - `cp .env` → staging to runtime path
   - `install.sh` → all packages
   - `git pull` → ensure latest code
   - `dry_run_gpu_cpu.py --n-seeds 2`
4. Polls every 30 seconds (reconnects each poll to avoid SSH timeout)
5. Prints final PASS/FAIL result

### Dry run checks (what PASS looks like)

```
INFO:__main__:=== GPU_CPU Dry Run (run_id=..., n_seeds=2) ===
INFO:__main__:  [PASS] ENV_KEYS
INFO:__main__:  [PASS] PLATFORM_LINUX  OS: Linux
INFO:__main__:  [PASS] GPU_AVAILABLE  NVIDIA A100-SXM4-80GB | 79.3 GB
INFO:__main__:  [PASS] TRANSFORMER_LENS_IMPORT  installed (v2.18.0+)
INFO:__main__:  [PASS] NNSIGHT_IMPORT  v0.7.0
INFO:__main__:  [PASS] FLASH_ATTENTION_IMPORT  v2.7.4.post1
INFO:__main__:  [PASS] OSM_LOAD_LLAMA_3.1_8B_INSTRUCT  attn=flash_attention_2
INFO:__main__:  [PASS] OSM_BATCH_INFERENCE_LLAMA_3.1_8B_INSTRUCT  2 responses, ...
INFO:__main__:  [PASS] OSM_LOAD_QWEN2.5_7B_INSTRUCT  attn=flash_attention_2
INFO:__main__:  [PASS] OSM_BATCH_INFERENCE_QWEN2.5_7B_INSTRUCT  2 responses, ...
INFO:__main__:  [PASS] OSM_LOAD_GEMMA.2.2B.IT  attn=flash_attention_2
INFO:__main__:  [PASS] OSM_BATCH_INFERENCE_GEMMA.2.2B.IT  2 responses, ...
INFO:__main__:  [PASS] OSM_LOAD_PHI.4.MINI.INSTRUCT  attn=flash_attention_2
INFO:__main__:  [PASS] OSM_BATCH_INFERENCE_PHI.4.MINI.INSTRUCT  2 responses, ...
INFO:__main__:  [PASS] CDVA_PATCHING_ONE_PAIR  delta_logit=0.xxxx
INFO:__main__:  [PASS] OUTLINES_CONSTRAINED_JSON  {"answer": ...}
```

### Monitoring a running session

```bash
ssh root@provider.a100.dsm.val.akash.pub -p 32355
# Password: MirageVM2026!

tmux attach -t full          # attach to pipeline session
tail -f /workspace/full_pipeline.log   # tail the log
```

---

## 12. Scripts Reference

All scripts live in `akash/`. Run from repo root.

| Script | Purpose |
|---|---|
| `_deploy_mirage.py` | Full Akash Console API deployment: SDL → bids → lease → SSH poll |
| `_full_pipeline.py` | **Primary entry point.** Upload .env + chain install + git pull + dry run |
| `predownload_models.py` | Pre-downloads all 4 OSM models to HF cache (called from install.sh) |
| `_poll_pipeline.py` | Poll `/workspace/full_pipeline.log` until PIPELINE_DONE |
| `_reinstall.py` | Re-run install.sh on running VM |
| `_quick_check.py` | Fast check: VM reachable? tmux sessions? workspace contents? |
| `_diagnose2.py` | Deep diagnostics: PID 1 uptime, cgroup memory, disk, host load |
| `_check_pkgs.py` | Verify all required packages via a Python script on VM |
| `install.sh` | Package installer (runs on VM). See §6 and §7. |
| `vm_ssh.txt` | VM host/port/dseq (gitignored, updated per deployment) |

### Current active VM

| Field | Value |
|---|---|
| DSEQ | `27071620` |
| SSH | `ssh root@provider.a100.dsm.val.akash.pub -p 32355` |
| Password | `MirageVM2026!` |
| RAM allocated | **64 GiB** (cgroup-verified: 68,719,476,736 bytes) |
| GPU | NVIDIA A100-SXM4-80GB, 79.3 GB VRAM |
| Cost | ~$1.25/hr |

---

## 13. Troubleshooting Index

### Memory / container restart issues (most common)

| Symptom | First thing to check | Root cause | Fix |
|---|---|---|---|
| Container restarts every few minutes during model loading | `cat /sys/fs/cgroup/memory.max` | SDL `memory:` too low (e.g. 16Gi) — cgroup OOM | Redeploy with `memory: 64Gi` |
| `free -h` shows 2 TiB but container still OOMs | Don't trust `free -h` in containers | `free` reads host RAM, not cgroup limit | Check `/sys/fs/cgroup/memory.max` instead |
| Container restarts even with 64 GiB cgroup, only 300 MB used | `cat /proc/loadavg` | Node-level eviction (host overloaded) | Reduce model sizes; switch to sequential load/unload |
| Log file disappears after restart | Ephemeral storage wiped | Container restarted → `/workspace` cleared | Use chained tmux session (see §8) |
| `memory.max` shows correct limit but OOM still happens | Check GPU kernel memory | Pinned GPU memory not tracked by cgroup | Reduce peak VRAM by using smaller models |

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
| `HUGGINGFACE_TOKEN` missing | `.env` not on VM | Upload via SFTP BEFORE launching install chain; see §10 |
| Packages gone after reconnect | Container restarted | Chain install+code in single tmux session; see §8 |
| `ModuleNotFoundError: dotenv` | Install exited early | Apply `__version__` fix to install.sh |
| OOM during CDVA patching | TL + HF model both in VRAM | Set threshold to 1.0× in `cdva_patching.py` |

### SSH / connectivity issues

| Symptom | Cause | Fix |
|---|---|---|
| `SSHException: SSH session not active` | Paramiko session timed out | Reconnect before each poll iteration |
| `UnicodeEncodeError` printing VM output | Windows cp1252 + pip progress bars | Wrap print() with `.encode("ascii", errors="replace")` |

---

## 14. Cost Estimate

| Phase | Duration | Cost at ~$1.25/hr |
|---|---|---|
| Install + 2-seed dry run | ~10 min | ~$0.21 |
| Full GPU behavioral eval (4 OSM models, full dataset) | ~6–10 hr | ~$7.50–$12.50 |
| CDVA patching (4 models) | ~3–5 hr | ~$3.75–$6.25 |
| CPU-only API eval | runs on separate machine | — |
| **Total (approximate)** | **~10–15 hr** | **~$12–$19** |

**Close the deployment when done:**
```python
import requests
requests.delete("https://console-api.akash.network/v1/deployments/27071620",
                headers={"x-api-key": AKASH_API_KEY})
```

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
| OSM-4 | `microsoft/Phi-4-mini-instruct` | 3.8B | ~8 GB | nnsight | Pending test |

**Why Gemma-2-9B was replaced with Gemma-2-2B:**
The 9B model caused node-level container evictions on Akash providers during loading due to high GPU pinned memory pressure (~18 GB VRAM peak). Gemma-2-2B uses the identical architecture (same TransformerLens hooks, same `-it` instruction format) with 4× less memory. Research coverage of the Gemma model family is preserved.

**Total VRAM across all 4 models:** ~42 GB (down from ~56 GB). All fit comfortably on one A100-80GB.
