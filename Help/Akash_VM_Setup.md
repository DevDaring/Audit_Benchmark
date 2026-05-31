# Akash GPU VM Setup — MIRAGE Project
### Complete Field Guide: Deployment, Package Installation, Flash Attention, and Running Code

*This document was written after working through every failure mode end-to-end.
Every "Issue" section describes something that actually broke and how it was fixed.*

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
9. [Uploading .env to the VM](#9-uploading-env-to-the-vm)
10. [Running the Dry Run and GPU Pipeline](#10-running-the-dry-run-and-gpu-pipeline)
11. [Scripts Reference](#11-scripts-reference)
12. [Troubleshooting Index](#12-troubleshooting-index)
13. [Cost Estimate](#13-cost-estimate)

---

## 1. What Akash Is

Akash Network is a decentralised cloud marketplace. Providers (GPU node operators) bid on your deployment request. You accept the cheapest bid and pay in AKT (Akash token). The billing is per-block (~6 seconds per block, ~$2–5/hr for an A100).

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
          units: "4"
        memory:
          size: 16Gi
        storage:
          - size: 200Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
  placement:
    westcoast:
      pricing:
        mirage:
          denom: uakt
          amount: 10000000

deployment:
  mirage:
    westcoast:
      profile: mirage
      count: 1
```

**What we learned about resource values:**
- `cpu: "4"` worked. Higher values reduced the number of providers that could bid.
- `memory: 16Gi` is sufficient for MIRAGE (models run on GPU not CPU RAM). Requesting more reduces bidders.
- `storage: 200Gi` is plenty.
- `gpu: units: 1` with just `vendor: nvidia:` (no model or RAM filter) gives the most bids.

---

## 5. Container Startup Sequence

After lease is accepted, the container starts and runs the startup command:

1. `apt-get install` — basic tools (git, curl, tmux, openssh-server, wget)
2. Set root password
3. Configure and start SSH daemon (`/usr/sbin/sshd`)
4. `git clone` or `git pull` the repo
5. Write `vm_ready.txt`
6. `tail -f /dev/null` — keeps PID 1 alive so container doesn't exit

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

**Why:** Ubuntu 22.04's system `python3-pip` is a patched version. When you upgrade pip in-place via `pip install --upgrade pip`, the pip module is replaced in `/usr/local/lib/python3.10/dist-packages/pip/` but the system Python's module discovery breaks. Result: `python3 -m pip` silently fails — returns `No module named pip`.

**The correct approach:** Bootstrap pip from `get-pip.py` FIRST, before any upgrades:
```bash
# GOOD — always start with this
apt-get install -y python3-dev       # dev headers, NOT python3-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py --quiet
python3 -m pip --version             # must succeed before proceeding
python3 -m pip install setuptools wheel packaging ninja
```

This guarantees `python3 -m pip` is always healthy.

### Issue 5: Packages installed but not importable

**Symptom:** `pip install torch` reports success, but `python3 -c "import torch"` gives `ModuleNotFoundError`.

**Cause:** There are often two Python executables:
- `/usr/bin/python3` — system Python (what SSH shells use)
- `/usr/local/bin/python3.10` — possibly installed separately

When pip installs to one Python's site-packages, the other Python can't see them.

**Fix:** Always use `python3 -m pip install` (not `pip3 install`). This guarantees packages go to the same Python that will run them. Verify with:
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
fails because `transformer_lens==2.18.0` (the last v2 stable release) has **no `__version__` attribute**.

This causes `set -e` to abort the script before the remaining packages (python-dotenv, scipy, etc.) are installed.

**Fix:** Never rely on `__version__` for packages that don't expose it:
```bash
# BAD
python3 -c "import transformer_lens; print('[install] transformer_lens', transformer_lens.__version__)"

# GOOD
python3 -c "import transformer_lens; print('[install] transformer_lens OK (v2.18.0 has no __version__ attr)')"
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

Flash Attention is notoriously difficult to install. This section documents exactly what works.

### Compatibility Matrix (what actually works)

| PyTorch | CUDA | Python | Flash Attn | ABI | Status |
|---|---|---|---|---|---|
| 2.6.0+cu124 | 12.4 | 3.10 | 2.7.4.post1 | cxx11abiFALSE | **WORKS ✅** |
| 2.5.x | 12.4 | 3.10 | 2.6.x | cxx11abiFALSE | Likely works |
| 2.6.0 | 12.4 | 3.12 | any | — | **Fails** — no cp312 wheel |
| 2.6.0 | 11.x | any | any | — | **Fails** — wheel is cu12 |

### Why `cxx11abiFALSE` matters

PyTorch wheels distributed via pip on Linux are compiled with the pre-C++11 ABI (`cxx11abiFALSE`). Flash Attention must be compiled with the **matching ABI**. The `cxx11abiTRUE` variant is only for PyTorch compiled from source.

### Prebuilt wheel URL (fastest, most reliable)

```bash
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/\
v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

python3 -m pip install "$FLASH_WHL" --no-deps
```

The `--no-deps` flag prevents pip from trying to install/upgrade torch to satisfy flash_attn's declared dependency, which would replace our pinned torch 2.6.0.

### Fallback: build from source

Only use this if the prebuilt wheel fails (e.g. new CUDA version):

```bash
MAX_JOBS=4 python3 -m pip install flash-attn --no-build-isolation
```

Expect ~45–60 minutes compile time on the VM.

### Verification

```bash
python3 -c "import flash_attn; print(flash_attn.__version__)"
# Expected: 2.7.4.post1
```

---

## 8. The Ephemeral Storage Problem

**This is the single most frustrating issue when using Akash containers.**

### What happens

Akash containers have **ephemeral storage only** (unless you explicitly request persistent storage with `class: beta3` in the SDL). The ephemeral filesystem is:
- Alive while the container is running
- **Wiped completely on container restart**

The container's startup command (`git clone ...`) re-populates `/workspace/Audit_Benchmark` from GitHub on every restart. But installed Python packages in `/usr/local/lib/python3.10/dist-packages/` are wiped.

### Why the container keeps restarting

Observed restarts happen approximately 1–5 minutes after the last active workload (tmux session) finishes. The Akash provider appears to restart idle containers under certain conditions (exact policy unknown — could be memory pressure, health checks, or provider-side scheduling).

**Evidence:**
- `uptime` shows 134 days (HOST kernel) — the host is stable
- `ps -p 1 -o etimes,comm` shows PID 1 (`tail -f /dev/null`) uptime of only 5 minutes — container restarted recently
- `/workspace` contents are always: `Audit_Benchmark/` (re-cloned) + `vm_ready.txt` (re-written by startup.sh)

### The solution: chain everything in one tmux session

**Never run install and then reconnect to run code.** Install + code must run in the same uninterrupted tmux session:

```bash
# WRONG: two separate steps — restart can happen between them
# Step 1: tmux "install" -> install.sh
# --- container may restart here ---
# Step 2: tmux "run" -> python3 dry_run.py   # packages are GONE

# CORRECT: one chained command
tmux new-session -d -s pipeline \
  'bash install.sh && cp /workspace/mirage.env /workspace/Audit_Benchmark/Code/mirage/.env && cd /workspace/Audit_Benchmark/Code/mirage && python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2; echo DONE'
```

**The cp step** copies a pre-uploaded `.env` file from `/workspace/mirage.env` (a staging path not in the git repo) to the runtime path. The `.env` must be uploaded via SFTP before launching the tmux chain (see §9).

### Persistent storage (better long-term fix)

For production runs, add persistent storage to the SDL:
```yaml
storage:
  - size: 200Gi       # ephemeral (root filesystem)
  - size: 100Gi       # persistent
    name: data
    mount: /data       # mounted at /data, survives restarts
    class: beta3       # beta3 = persistent SSD on Akash
```

Then install packages to the persistent path:
```bash
python3 -m pip install --target /data/site-packages torch ...
export PYTHONPATH=/data/site-packages:$PYTHONPATH
```

Note: persistent storage costs extra and reduces the number of providers that will bid.

---

## 9. Uploading .env to the VM

The `.env` file contains API keys and must never be committed to git. It must be uploaded to the VM after every container restart.

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
sftp.put(str(LOCAL_ENV), "/workspace/mirage.env")   # staging path
sftp.close()
```

Then in the tmux chain, copy it to the runtime path:
```bash
cp /workspace/mirage.env /workspace/Audit_Benchmark/Code/mirage/.env
```

**Why the staging path?** The runtime path `/workspace/Audit_Benchmark/Code/mirage/.env` is inside the git repo. After a container restart and `git pull`, git might overwrite this file (if `.env` is accidentally tracked). The staging path `/workspace/mirage.env` is outside the repo and safer.

### Alternative: encode .env in the SDL environment variable

You can base64-encode the entire `.env` and pass it in the SDL `env:` block:
```bash
# On local machine
python3 -c "import base64,pathlib; print(base64.b64encode(pathlib.Path('Code/mirage/.env').read_bytes()).decode())"
```

Then in SDL:
```yaml
env:
  - MIRAGE_ENV_B64=<base64_string>
```

And in startup.sh:
```bash
echo "$MIRAGE_ENV_B64" | base64 -d > /workspace/Audit_Benchmark/Code/mirage/.env
```

**Caveat:** This makes the `.env` contents visible in the SDL YAML which is submitted to the Akash network. Treat this with caution for production API keys.

---

## 10. Running the Dry Run and GPU Pipeline

### Dry run with 2 seeds (pipeline validation)

```bash
# On VM (inside tmux):
cd /workspace/Audit_Benchmark/Code/mirage
PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage \
python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2
```

The `--n-seeds 2` flag (added in our parallelism update) processes exactly 2 probe seeds through the full GPU pipeline:
1. ENV_KEYS — all API keys present
2. PLATFORM_LINUX — Linux x86_64 required for flash-attention
3. GPU_AVAILABLE — CUDA and A100 detected
4. TRANSFORMER_LENS_IMPORT — TL 2.18.0
5. NNSIGHT_IMPORT — nnsight 0.7.0
6. FLASH_ATTENTION_IMPORT — flash_attn 2.7.4.post1
7. OSM_BATCH_INFERENCE — loads all 4 OSM models, runs batched forward pass on 2 seeds
8. CDVA_PATCHING — activation patching with TransformerLens/nnsight
9. OUTLINES_CONSTRAINED_JSON — constrained JSON decoding

### Full pipeline (from local machine via `akash/_full_pipeline.py`)

```bash
# From repo root:
python akash/_full_pipeline.py
```

This script:
1. Uploads `.env` to `/workspace/mirage.env` via SFTP
2. Kills any stale tmux sessions
3. Launches: `install.sh && cp .env && dry_run_gpu_cpu.py --n-seeds 2`
4. Polls every 30 seconds (reconnecting each poll)
5. Prints final result

### Monitoring a running session

```bash
# SSH in:
ssh root@provider.a100.dsm.val.akash.pub -p 30594
# Password: MirageVM2026!

# Attach to running tmux:
tmux attach -t full

# Or tail the log:
tail -f /workspace/full_pipeline.log
```

### What "PASS" looks like in the dry run

```
INFO:__main__:=== GPU_CPU Dry Run (run_id=..., n_seeds=2) ===
INFO:__main__:  [PASS] ENV_KEYS
INFO:__main__:  [PASS] PLATFORM_LINUX  OS: Linux
INFO:__main__:  [PASS] GPU_AVAILABLE  NVIDIA A100-SXM4-80GB | 79.3 GB
INFO:__main__:  [PASS] TRANSFORMER_LENS_IMPORT  v... (TL has no __version__)
INFO:__main__:  [PASS] NNSIGHT_IMPORT  v0.7.0
INFO:__main__:  [PASS] FLASH_ATTENTION_IMPORT  v2.7.4.post1
INFO:__main__:  [PASS] OSM_BATCH_INFERENCE_LLAMA_3_1_8B_INSTRUCT  2 responses, ...
...
```

---

## 11. Scripts Reference

All scripts live in `akash/`. Run from repo root.

| Script | Purpose |
|---|---|
| `_deploy_mirage.py` | Full Akash Console API deployment: SDL → bids → lease → SSH poll |
| `_full_pipeline.py` | **Primary entry point.** Upload .env + chain install + dry run |
| `_install_and_dryrun.py` | Like `_full_pipeline.py` but without the .env staging step |
| `_poll_pipeline.py` | Poll `/workspace/full_pipeline.log` until PIPELINE_DONE |
| `_reinstall.py` | Re-run install.sh on running VM (uses get-pip.py bootstrap) |
| `_quick_check.py` | Fast check: is VM reachable? tmux sessions? workspace contents? |
| `_diagnose2.py` | Deep diagnostics: PID 1 uptime, memory, storage, dist-packages |
| `_check_pkgs.py` | Verify all required packages via a Python script on VM |
| `install.sh` | Package installer (runs on VM). See §6 and §7 for details. |
| `vm_ssh.txt` | VM host/port/dseq (gitignored, updated per deployment) |

### Environment variables used by scripts

| Var | Default | Description |
|---|---|---|
| `AKASH_VM_HOST` | `provider.a100.dsm.val.akash.pub` | SSH host |
| `AKASH_VM_PORT` | `30594` | SSH port |
| `AKASH_VM_USER` | `root` | SSH username |
| `AKASH_VM_PASSWORD` | `MirageVM2026!` | SSH password |

Set as PowerShell env vars before running:
```powershell
$env:AKASH_VM_HOST = "provider.a100.dsm.val.akash.pub"
$env:AKASH_VM_PORT = "30594"
$env:AKASH_VM_PASSWORD = "MirageVM2026!"
python akash/_full_pipeline.py
```

---

## 12. Troubleshooting Index

### Deployment issues

| Symptom | Cause | Fix |
|---|---|---|
| No bids after 5 min | Price too low / GPU over-constrained | Raise `amount` to 10 000 000; remove `ram:`, `host:`, `signedBy:` |
| `{"code":"invalid_type","path":["manifest"]}` | Passing SDL text as manifest | Extract `data.manifest` from POST /v1/deployments response; pass that blob |
| SSH connection refused | Container still booting | Wait 2–3 min after lease accepted |
| Authentication failed | Wrong SSH key / password | Use password auth with `MirageVM2026!`; set `PasswordAuthentication yes` in sshd_config |

### Package installation issues

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pip` | Ubuntu apt pip upgraded in-place | Bootstrap from `get-pip.py` instead (see §6) |
| `import torch` fails after install | pip installed to different Python | Always use `python3 -m pip`, verify with `python3 -m pip show torch` |
| `flash_attn` compile fails | Wrong ABI, torch, or CUDA version | Use prebuilt wheel for cp310+cu124+torch2.6+cxx11abiFALSE (see §7) |
| `transformer_lens.__version__` AttributeError | TL 2.18.0 has no `__version__` | Change verification to `print("OK")` instead of `print(.__version__)` |
| `set -e` aborts install mid-way | Any verification command exits non-zero | Check ALL `python3 -c` verification lines; use `|| true` for known-OK cases |

### Runtime issues

| Symptom | Cause | Fix |
|---|---|---|
| `HUGGINGFACE_TOKEN` missing | `.env` not on VM (wiped by restart) | Upload via SFTP before starting pipeline; see §9 |
| Packages gone after reconnect | Akash container restarted | Chain install+code in single tmux session; see §8 |
| `ModuleNotFoundError: dotenv` | Install exited early (transformer_lens fix not applied) | Apply `__version__` fix to install.sh; see Issue 6 |
| OOM during CDVA patching | TL + HF model both in VRAM | Set threshold to 1.0× in `cdva_patching.py` (already fixed for 80 GB) |
| Batch inference gives empty output | `padding_side` not set | Ensure `tokenizer.padding_side = "left"` before batch tokenization |

### SSH / connectivity issues

| Symptom | Cause | Fix |
|---|---|---|
| `SSHException: SSH session not active` | Paramiko session timed out during long sleep | Reconnect before each poll: `client = conn()` at the start of every loop iteration |
| `TimeoutError: [WinError 10060]` | Container temporarily unreachable | Wait 60 s and retry; Akash providers occasionally have brief network blips |
| `UnicodeEncodeError` printing VM output | Windows cp1252 console + pip progress bars | Wrap every `print()` in a try/except that falls back to `encode("ascii", errors="replace")` |

---

## 13. Cost Estimate

| Phase | Duration | Cost at ~$3/hr |
|---|---|---|
| Install + 2-seed dry run | ~30 min | ~$1.50 |
| Full GPU behavioral eval (4 OSM models, full dataset) | ~8–12 hr | ~$24–$36 |
| CDVA patching (4 models) | ~4–6 hr | ~$12–$18 |
| CPU-only API eval | runs in parallel on separate machine | — |
| **Total (approximate)** | **~15–20 hr** | **~$45–$65** |

**Close the deployment when done to stop billing:**
```bash
# Via Akash Console UI, or API:
DELETE https://console-api.akash.network/v1/deployments/{dseq}
```

---

## Appendix: Version Pins That Are Known to Work

```
Python:           3.10.12  (system, Ubuntu 22.04 in nvidia/cuda:12.4.1 image)
CUDA:             12.4.1
torch:            2.6.0+cu124
torchvision:      (latest matching)
flash_attn:       2.7.4.post1  (prebuilt wheel, cxx11abiFALSE)
transformer_lens: 2.18.0
nnsight:          0.7.0
transformers:     4.57.6
accelerate:       1.13.0
outlines:         (latest)
pandas:           2.0.3  (pinned by transformer_lens)
numpy:            1.26.4 (pinned by transformer_lens)
python-dotenv:    1.2.2
```

GPU achieved: **NVIDIA A100-SXM4-80GB, 79.3 GB VRAM**
DSEQ of working deployment: `27070733`
Provider: `provider.a100.dsm.val.akash.pub`
SSH port: `30594`
Password: `MirageVM2026!`
