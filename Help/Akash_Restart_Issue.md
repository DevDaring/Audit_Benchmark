# Akash Network — Container Restart Problem Report

**Project:** MIRAGE — Multi-Instance Reasoning Audit for Generative Evaluation  
**Goal:** Run a GPU inference pipeline (4 × LLM models, ~42 GB VRAM total) on an Akash A100-80GB VM  
**Status:** Blocked by repeated container evictions. Seeking expert input.  
**Date:** 2026-05-31

---

## 1. What We Are Trying to Do

Run a research pipeline on a single Akash GPU container:

1. Install Python packages (~150 s, no GPU)
2. Pre-download 4 HuggingFace models to disk cache (~42 GB total, no GPU)
3. Run a 2-seed dry run that loads each model sequentially into GPU VRAM, runs inference, and unloads

**Models used:**

| Model | Params | VRAM (bf16) | CPU RAM peak during load |
|---|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16 GB | ~14 GB |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~14 GB | ~12 GB |
| `google/gemma-2-2b-it` | 2B | ~4 GB | ~4 GB |
| `microsoft/Phi-4-mini-instruct` | 3.8B | ~8 GB | ~7 GB |

**Container spec requested in SDL:**

```yaml
cpu:     4 units
memory:  64Gi
storage: 200Gi (ephemeral)
gpu:     1 × NVIDIA (any)
```

**Actual provider assigned:** `provider.a100.dsm.val.akash.pub`  
**GPU allocated:** NVIDIA A100-SXM4-80GB (79.3 GB VRAM, confirmed via `nvidia-smi`)  
**DSEQ:** 27071620  
**SSH port:** 32355

---

## 2. The Problem

**The container restarts unexpectedly and repeatedly, always within 10–11 minutes of container boot.**

Each restart:
- Terminates all running processes (tmux, Python, downloads)
- Wipes all installed packages (ephemeral storage — no persistence)
- Forces us to re-install everything from scratch (~150 s minimum before any work can resume)
- Wastes GPU rental cost

We have never completed the full dry run on this provider. The furthest we got was 2 out of 4 models passing (LLaMA + Qwen), after which the container restarted.

---

## 3. Observed Symptoms

### 3.1 `free -h` shows 2 TiB — completely misleading

```
               total        used        free
Mem:           2.0Ti        81Gi       1.2Ti
```

This is the **host machine's total RAM** (shared across all tenants). It is NOT our container's allocated RAM. `free -h` reads `/proc/meminfo` which is a host-level interface invisible to cgroups.

**Actual container RAM (from cgroup):**
```bash
$ cat /sys/fs/cgroup/memory.max
68719476736       # = 64 GiB (our SDL-requested memory: 64Gi)

$ cat /sys/fs/cgroup/memory.current
265740288         # = ~253 MB (actual current usage at idle)
```

Our container's RAM usage is always well under 64 GiB at the moment of restart. We are NOT hitting our own cgroup limit.

### 3.2 Host load average is very high

```bash
$ cat /proc/loadavg
14.20 10.53 10.43
```

Load average of 14 on a system means ~14 active/waiting threads at all times. This is a heavily loaded server running many tenants simultaneously.

### 3.3 Container PID 1 uptime is always short after a restart

```bash
$ ps -p 1 -o etimes=
428           # seconds = 7 minutes since last boot
```

After every eviction, PID 1 (`tail -f /dev/null` — our keepalive) restarts. The host kernel uptime (`uptime` command) shows 134 days — the HOST is stable. Only our container process is being restarted.

### 3.4 Restart happens at a consistent total container age

Across all observed runs on this provider, the container was evicted at approximately **10–12 minutes from container boot**, regardless of what we were doing at that moment:

| Run | Container age at start of our pipeline script | Age when evicted | Activity at eviction |
|---|---|---|---|
| 479397 (prev) | ~5 min | ~10.5 min | Qwen2.5-7B HF download started |
| 848403 (latest) | ~7 min | ~10.5 min | LLaMA-3.1-8B HF download started (predownload phase) |

In an earlier successful run (875362), the container survived ~15+ minutes — but that was on what appears to have been a less loaded provider state, and even then it was evicted when `gemma-2-9b-it` (18 GB) started downloading.

### 3.5 No error message visible to tenant

The container restart gives no error from our side. The pipeline log file simply disappears:
```
[240s]  tail: cannot open '/workspace/full_pipeline.log': No such file or directory
```

The Akash tenant cannot see the pod exit code or reason. Only the provider operator can see `kubectl describe pod` output, which would show `OOMKilled` (exit code 137) or `Evicted`.

---

## 4. What We Have Tried (Chronologically)

### Attempt 1 — SDL with `memory: 16Gi`

**What:** Initial deployment with 16 GiB RAM.  
**Result:** Container restarted immediately when LLaMA (8B, ~14 GB CPU RAM peak during loading) started loading. This is a container-level OOM (cgroup limit hit).  
**Learning:** 16 GiB is too small. HuggingFace Big Model Inference temporarily allocates the full model weight into CPU RAM during loading even though the final model resides in GPU VRAM.

---

### Attempt 2 — SDL with `memory: 64Gi`

**What:** Increased container RAM to 64 GiB.  
**Result:** Container no longer hits its OWN cgroup limit. However, it still restarts, now at the ~10-11 minute mark.  
**Learning:** The restart is no longer a container-OOM. It is a **host-level (node-level) eviction** — the kubelet on the provider's Kubernetes node is terminating our pod due to host-wide resource pressure.

---

### Attempt 3 — Parallel model loading (keep all 4 in VRAM simultaneously)

**What:** `load_all_osm_models()` loaded all 4 models at startup (~42 GB total).  
**Result:** During loading of model 3, the host evicted the container.  
**Learning:** Loading 4 models accumulates ~42 GB VRAM + multiple large CPU RAM peaks → drives up host pinned memory → triggers eviction faster.

---

### Attempt 4 — Sequential load/unload between models

**What:** Modified `dry_run_gpu_cpu.py` to load one model, test it, call `unload_model()` (which calls `del model`, `gc.collect()`, `torch.cuda.empty_cache()`), then load the next.  
**Result:** Peak VRAM at any time = ~16 GB (one model) instead of ~42 GB (all models). Container survived longer — LLaMA loaded and passed. But evicted when Qwen download started.  
**Learning:** Sequential loading reduces VRAM pressure but does not fix host-level eviction.

---

### Attempt 5 — Replace `gemma-2-9b-it` (9B) with `gemma-2-2b-it` (2B)

**What:** Switched from Gemma-2-9B (~18 GB VRAM, ~18 GB download) to Gemma-2-2B (~4 GB VRAM, ~4 GB download). Updated `config.py`, `predownload_models.py`, `README.md`, `MIRAGE_MASTER_PROMPT.md`, codemap.  
**Result:** Reduced total model footprint from ~56 GB to ~42 GB. But the container was still evicted during Qwen download.  
**Learning:** Even 14 GB downloads trigger eviction. The problem is not the specific model size — it is sustained high-bandwidth I/O on an overloaded host.

---

### Attempt 6 — Pre-download all models BEFORE GPU dry run

**Hypothesis:** Downloads happening while CUDA is active (GPU context initialized) might be triggering host pinned-memory allocation spikes. Separating downloads (pure I/O, no GPU) from GPU loading might prevent eviction.

**What:** Added `predownload_models.py` to the pipeline chain, called AFTER `install.sh` and BEFORE `dry_run_gpu_cpu.py`. `snapshot_download` downloads all 4 model repos to HF disk cache. No GPU activity during this phase.

**Pipeline chain (as of commit `c99c248`):**
```
install.sh  →  INSTALL_OK  →  git pull  →  cp .env
  →  predownload_models.py  (pure I/O, CUDA idle)
  →  PREDOWNLOAD_OK
  →  dry_run_gpu_cpu.py  (loads from disk cache, no network)
```

**Result:** Container was evicted DURING the predownload phase (during LLaMA download, no GPU active at all). This disproves the "download + CUDA = eviction" hypothesis. The eviction is happening at ~10-11 minutes of container age regardless of GPU activity.

**Exact log at eviction time (run 848403):**
```
[150s] INSTALL_OK — git pull done — predownload starting
[150s] [predownload] meta-llama/Llama-3.1-8B-Instruct ...
[150s] Fetching 17 files: 6%|  | 1/17
[180s] reconnect: timed out
[210s] tail: cannot open '/workspace/full_pipeline.log'  ← EVICTED
```

Container age at script start: ~7 minutes.  
Container age at eviction: ~7 + 3.5 = **~10.5 minutes from boot**.  
GPU state at eviction: **completely idle** (nvidia-smi would show 0 MiB used).

---

### Attempt 7 — Verify actual cgroup limits (diagnostic)

**Commands run on live container:**
```bash
cat /sys/fs/cgroup/memory.max          → 68719476736 (64 GiB — our limit)
cat /sys/fs/cgroup/memory.current      → 265740288 (253 MB — our usage)
cat /proc/loadavg                      → 14.20 10.53 10.43
ps -p 1 -o etimes=                     → 428 (7 minutes alive)
nvidia-smi --query-gpu=memory.used     → 0 MiB
```

**Conclusion:** At the moment we checked (after the latest restart), the container was using only 253 MB of its 64 GiB limit. There is NO container-level OOM. The host load is 14+ (very high).

---

## 5. Working Theory

The evictions are **node-level Kubernetes evictions**, not container-level OOM kills. The Kubernetes kubelet on the provider's node is evicting our pod to reclaim resources for the node itself.

Relevant Akash support issue: [https://github.com/akash-network/support/issues/246](https://github.com/akash-network/support/issues/246) (open as of 2026-05-31) — multiple tenants report containers being evicted with no clear cause, suspected to be provider-side resource pressure.

### Why the ~10-11 minute pattern

When the container boots, it first runs the startup command (git clone, sshd setup — ~60-90 s). Then our pipeline script connects and starts `install.sh`. The install downloads ~800 MB of Python packages at 100-150 MB/s. By minute 3-4 of the container's life, it starts doing large HuggingFace downloads (800 MB for PyTorch, then model weights). 

The sustained high-bandwidth I/O drives up host-level kernel buffer usage. On an overloaded node (load avg 14), this pushes the host past its own OOM threshold, triggering the kubelet to evict pods. Our container has been alive the longest by that point → it gets evicted first.

### Why run 875362 survived longer

The one run that got LLaMA + Qwen passing (earlier session) likely coincided with a period of lower host load. Load average fluctuates and a lower-load window allowed our container to survive for 15+ minutes.

---

## 6. Current State

- Container is alive (PID 1 uptime ~7-8 minutes at time of writing, 18:30 UTC)
- Host load average: 14.20 (high but potentially fluctuating)
- Our pipeline has NOT been run since the latest restart — we are waiting before launching to ensure maximum remaining uptime window
- Deployment DSEQ 27071620 is still active and billing at ~$1.25/hr

---

## 7. Things We Have NOT Yet Tried

### 7a. Request persistent storage (SDL `class: beta3`)

```yaml
storage:
  - size: 200Gi       # ephemeral root
  - size: 100Gi
    name: data
    mount: /data
    class: beta3      # persistent SSD — survives container restart
```

If packages were installed to `/data/site-packages` and models downloaded to `/data/hf_cache`, a container restart would not lose any work. The pipeline could resume from where it left off.

**Obstacle:** `class: beta3` reduces the number of providers that bid (most don't offer it). May need to raise `amount` further or accept no bids.

### 7b. Deploy to a different provider

The current provider (`provider.a100.dsm.val.akash.pub`) is consistently overloaded. A different provider might have lower host load and more stable container lifetimes.

**How:** Set `signedBy` in SDL to target a specific provider, or just redeploy and accept a different bid.

**Obstacle:** No way to filter for "providers with stable container lifetimes" from the SDL.

### 7c. Request more CPU memory to reduce pressure

Requesting `memory: 128Gi` instead of `64Gi` might deter the kubelet from evicting us (pods with higher memory requests are treated as higher-priority in some scheduler configs).

**Obstacle:** Fewer providers can bid on 128 GiB requests.

### 7d. Use a Kubernetes `priorityClass` or `QoS` setting

In Kubernetes, pods with `Guaranteed` QoS (requests == limits for all resources) are the last to be evicted. Akash SDL may support equivalent settings.

**Obstacle:** Unsure if Akash SDL exposes QoS or priorityClass control.

### 7e. Split the workload into two shorter containers

- Container A: Install + predownload all models (no GPU, ~10 min, then exits cleanly)
- Container B: Load models from persistent storage and run dry run / full pipeline

**Obstacle:** Requires persistent storage between containers, which needs `class: beta3` and a persistent volume claim.

### 7f. Use a provider that offers dedicated (bare-metal) access

Some Akash providers offer bare-metal GPU servers where the container is the only tenant. These do not have the multi-tenant memory pressure problem.

**How to identify:** Look for providers with `dedicated: true` in their attributes, or use Akash Console to filter.

---

## 8. Full Timeline of Observed Restarts on This Provider

| Time (UTC) | Container boot | Script start | Evicted at | Duration alive | Activity at eviction |
|---|---|---|---|---|---|
| ~18:00 | ~18:00 | ~18:05 | ~18:10–18:14 | ~10–14 min | Gemma-2-9B download |
| ~18:14:30 | ~18:14:30 | ~18:10:01 (script) | ~18:14:30 | — | Qwen download (479397) |
| ~18:14:30 | ~18:14:30 | ~18:21:45 | ~18:25:00 | ~10.5 min | LLaMA predownload (848403) |
| ~18:25:00 | ~18:25:00 | — | (still alive at 18:30) | ~5 min alive, ~5 min left | Idle |

---

## 9. Questions for the Expert

1. **Is there a way in Akash SDL to request a `Guaranteed` QoS pod** (cpu.limits == cpu.requests, memory.limits == memory.requests) to prevent kubelet from evicting us?

2. **Is `class: beta3` persistent storage reliable on Akash?** How much does it reduce the number of bidding providers? Is there a provider filter to only see providers that offer it?

3. **Is there a provider-level SLA or "dedicated" flag** that ensures our container is the only tenant on the node, eliminating multi-tenant memory pressure?

4. **What is the correct way to set node affinity or pod priority in Akash SDL** to reduce eviction risk?

5. **Is the high load average (14.20) expected for a shared GPU node** or is this provider specifically overloaded? How do we identify and select low-load providers before bidding?

6. **Can Akash containers use `emptyDir: { medium: Memory }` or similar Kubernetes features** to improve I/O performance during large downloads without increasing host memory pressure?

7. **Is there a known workaround for the ~10-minute eviction pattern** on specific providers? Some Akash community members might have encountered this.

8. **Is it possible to get a refund for the lease time lost to provider-side evictions?** The GPU was allocated but our workload could not run.

---

## 10. Codebase Context

- GitHub: `https://github.com/DevDaring/Audit_Benchmark` (public)
- Pipeline entry: `python akash/_full_pipeline.py` (from repo root)
- Install script: `akash/install.sh`
- Model predownload: `akash/predownload_models.py`
- Dry run: `Code/mirage/Dry_Run/dry_run_gpu_cpu.py --n-seeds 2`
- SDL generator: `akash/_deploy_mirage.py`

**Packages installed (pinned versions that are known to work):**
```
torch==2.6.0+cu124
flash_attn==2.7.4.post1 (prebuilt wheel, cxx11abiFALSE)
transformer_lens==2.18.0
nnsight==0.7.0
transformers>=4.47.0
accelerate>=0.34.0
```

---

## 11. Appendix — Exact SDL That Got Bids and Was Used

```yaml
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
          size: 64Gi
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

---

## 12. Appendix — Key Diagnostic Commands

Run these immediately after SSH-ing into the container:

```bash
# 1. Real container RAM limit (NOT free -h)
python3 -c "
limit = int(open('/sys/fs/cgroup/memory.max').read())
used  = int(open('/sys/fs/cgroup/memory.current').read())
print(f'Limit: {limit/1024**3:.1f} GiB  Used: {used/1024**3:.2f} GiB')
print('(free -h would show host RAM = wrong)')
"

# 2. How long this container instance has been alive
ps -p 1 -o etimes=
# < 60s = just booted, full window available
# > 600s = already past 10-min threshold, likely to be evicted soon

# 3. Host load (shared with all other tenants)
cat /proc/loadavg
# < 4.0  = healthy host
# 4.0-8.0 = moderate load, some risk
# > 8.0  = high load, elevated eviction risk
# > 12.0 = very high, eviction imminent

# 4. GPU state
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

# 5. Disk space
df -h /workspace
```
