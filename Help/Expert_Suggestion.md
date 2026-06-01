# MIRAGE — Akash Eviction Fix & Implementation Guide

**For:** Cursor (agentic implementation against `DevDaring/Audit_Benchmark`)
**Goal:** Stop losing all work to ~10-min container evictions on Akash GPU leases.
**Date:** 2026-06-01

> **How confident is this doc?** The Akash/Kubernetes facts below were verified
> against Akash docs and the Akash `support` issue tracker (links inline). The
> *root-cause* itself is a calibrated inference, not a certainty — only the
> provider operator can see `kubectl describe pod` and read the exact eviction
> reason. **Step 0 below adds the instrumentation that will confirm it.** The
> fix in P1, however, works whether the cause is disk *or* memory, so you do not
> need to wait for confirmation to start implementing.

---

## 0. The corrected diagnosis (read this first)

Your report concluded: *node-level eviction caused by host memory pressure from
download buffers.* The **node-level** part is almost certainly right. The
**memory** part is probably wrong. Here is the re-read of your own evidence:

| Your observation | What it actually implies |
|---|---|
| Every eviction happened *during a model download* (Gemma / Qwen / LLaMA) | Correlates with **disk writes**, not wall-clock time. The "~10 min" is just when your script reaches the heavy-download phase. |
| `memory.current` = 253 MB of 64 GiB at restart | You are nowhere near a memory limit. Memory is **not** the bottleneck. |
| `free -h` shows 1.2 TiB free on host | Host has enormous RAM headroom; host memory-pressure eviction is implausible. |
| "Idle container will die at 10 min too" | This is a **prediction in your report, not a logged eviction.** Every *actual* eviction was during I/O. |
| No error visible to tenant; log file just vanishes | Matches an **Evicted** pod exactly. |

### Why disk, not memory

- **Page cache is reclaimable.** Linux drops clean cache pages under pressure
  instead of OOM-killing. "Downloads fill kernel buffers → host OOM" is not how
  the kernel behaves when there is 1.2 TiB free.
- **On Akash, exceeding ephemeral storage = exit 137 = "Evicted"** — visually
  identical to an OOM kill from the tenant side. This is a documented Akash
  behavior, not a theory:
  - Akash support issue #42 — a pod whose storage `request == limit`, when it
    writes past that limit, restarts with `reason: "Evicted"` and the note
    *"Container exceeded its local ephemeral storage limit"*, exit code 137.
    <https://github.com/akash-network/support/issues/42>
- **Ephemeral-storage eviction is independent of memory pressure**, and the QoS
  class calculation **excludes** `ephemeral-storage`. So a "Guaranteed QoS" pod
  is **still evicted** when it (or the node) runs out of disk.
  <https://www.sysdig.com/blog/kubernetes-pod-evicted>
  <https://jorijn.com/en/knowledge-base/kubernetes/storage/kubernetes-ephemeral-storage-limits-and-eviction/>

### Two disk sub-cases (the fix covers both)

1. **Node-level disk pressure (`nodefs`/`imagefs`).** The provider's node disk
   fills up from *all* tenants combined. The kubelet evicts pods ranked by disk
   usage — and you, downloading 42 GB, are the fattest target. Fits "evicted
   during downloads" perfectly.
2. **Your pod's own ephemeral limit.** Your HF cache defaults to
   `/root/.cache/huggingface` (ephemeral root FS), and the
   `nvidia/cuda:...-devel` image writable layer is large. If the provider hands
   you less ephemeral than the 200Gi you asked for (overcommit), you trip your
   own limit.

**The single thing you never measured:** `df -h /workspace` *during* a download.
Step 0 fixes that.

---

## 1. Priority order

| Pri | Change | Why | Effort |
|---|---|---|---|
| **P0** | Persistent **watchdog log** that survives eviction | Confirms disk-vs-memory; captures the data you're missing | 1 file |
| **P1** | **Persistent volume + resumable pipeline** | Makes eviction *harmless*; works regardless of cause | core |
| **P2** | Shrink ephemeral/disk footprint (runtime image, caches → persistent, downloads → persistent) | Directly removes the disk-eviction trigger | medium |
| **P3** | **Better provider + right-sized GPU** | Gets you off the overloaded A100 host entirely | low |
| **P4** | (Optional) Prebuilt Docker image | Zero install time; gold-standard robustness | medium |

---

## P0 — Watchdog that survives the restart (do this first)

The reason you can't diagnose this is your log lives on ephemeral storage and
dies with the pod. Put the log on the **persistent volume** (added in P1) so
that immediately after the next eviction you can `tail` it and see exactly what
spiked.

**Create `akash/watchdog.sh`:**

```bash
#!/usr/bin/env bash
# Appends a resource snapshot every 10s to a log on PERSISTENT storage,
# so it survives pod eviction. After a restart: tail -50 /data/logs/watchdog.log
mkdir -p /data/logs
LOG=/data/logs/watchdog.log
echo "=== watchdog boot $(date -u +%FT%TZ) pid1_age=$(ps -p 1 -o etimes= 2>/dev/null) ===" >> "$LOG"
while true; do
  ts=$(date -u +%FT%TZ)
  mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  memmax=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
  load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)
  # EPHEMERAL disk (the suspected culprit) and its inodes:
  eph=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $3"/"$2" ("$5")"}')
  ephi=$(df -i /workspace 2>/dev/null | awk 'NR==2{print $5}')
  # PERSISTENT disk:
  dat=$(df -h /data 2>/dev/null | awk 'NR==2{print $3"/"$2" ("$5")"}')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
  age=$(ps -p 1 -o etimes= 2>/dev/null)
  echo "$ts pid1_age=${age}s mem=${mem}/${memmax} load=[${load}] eph=${eph} eph_inodes=${ephi} data=${dat} gpu=${gpu}" >> "$LOG"
  sleep 10
done
```

**What it proves after the next eviction:**

- Last line shows `eph` near 100% (or `eph_inodes` near 100%) → **disk eviction
  confirmed.** Move on with P1/P2 confidently.
- Last line shows `mem` climbing toward `memmax` → it really is memory; revisit
  the memory strategy.
- Neither spikes but pod still died → node-level pressure from *other* tenants
  (still solved by P1 making you a low-priority eviction target + resumability).

> **Definitive proof (optional):** politely ask the provider operator to run
> `kubectl describe pod` / check `lease-events` — the eviction message names the
> exact resource (`memory` vs `ephemeral-storage`). The watchdog gets you 90% of
> the way without needing them.

---

## P1 — Persistent volume + resumable pipeline (the core fix)

**Principle:** if you can't reliably prevent eviction, make it cost ~30 seconds
instead of starting from zero. Put everything expensive on a persistent volume
and make the pipeline skip already-completed steps.

### Why this works regardless of root cause

- **Persistent storage survives container restarts within the same lease.** It
  is only lost on lease *close*, deployment *update*, or *migration* — none of
  which a simple eviction-restart triggers.
  <https://github.com/akash-network/docs/blob/master/readme/stack-definition-language.md>
- **PVC writes do not count against `ephemeral-storage`.** Persistent volumes
  live on separate dedicated drives, excluded from the kubelet's ephemeral
  accounting — so moving the 42 GB of downloads onto `/data` removes the disk
  pressure that is (probably) causing the eviction, *and* makes you a poor
  eviction target under node disk pressure.
  <https://jorijn.com/en/knowledge-base/kubernetes/storage/kubernetes-ephemeral-storage-limits-and-eviction/>
- Even in the worst case (memory, or other-tenant pressure), a resumable
  pipeline completes across N restart cycles instead of never.

### 1a. New SDL (have `akash/_deploy_mirage.py` emit this)

Key changes vs your Appendix-11 SDL: a **persistent `/data` volume**, **smaller
ephemeral root** (models no longer live there), **env vars redirecting all
caches to `/data`**, and a **supervised, resumable command**.

```yaml
version: "2.0"
services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04   # see P2: runtime image is smaller
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
      # --- redirect ALL heavy writes to the persistent volume ---
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
        set -uo pipefail
        # fast, re-runs each boot (these land on ephemeral root, ~15s)
        apt-get update -qq && apt-get install -y git curl tmux openssh-server wget python3-venv > /dev/null 2>&1
        echo "root:${ROOT_PASSWORD}" | chpasswd
        mkdir -p /run/sshd
        sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
        sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
        /usr/sbin/sshd
        # persistent dirs
        mkdir -p /data/logs /data/state /data/hf_cache /data/pip_cache /data/cache /workspace
        # clone/update repo ONTO persistent storage so it survives evictions too
        if [ ! -d "${REPO_DIR}/.git" ]; then
          git clone "${GITHUB_REPO}" "${REPO_DIR}" || true
        else
          git -C "${REPO_DIR}" pull --ff-only || true
        fi
        # start the survivable watchdog + the resumable pipeline supervisor
        nohup bash "${REPO_DIR}/akash/watchdog.sh"          >> /data/logs/watchdog_boot.log 2>&1 &
        nohup bash "${REPO_DIR}/akash/supervise_pipeline.sh" >> /data/logs/supervise.log     2>&1 &
        echo "VM_READY $(date -u +%FT%TZ)" > /workspace/vm_ready.txt
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
          - size: 30Gi                 # ephemeral root: image layer + apt + logs only
          - name: data                 # persistent: venv + 42GB models + state + logs
            size: 120Gi
            attributes:
              persistent: true
              class: beta3             # NVMe; fall back to beta2 (SSD) if no bids — see P3
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:                  # any GPU; see P3 about right-sizing
    mirage-svc-mount: {}               # (placeholder so you remember the params block below)
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

**You must also add the volume mount in the service** (Akash links a named
storage profile to a mount via `params.storage`). Add this under
`services.mirage` (it can sit alongside `image`, `env`, `command`, `expose`):

```yaml
    params:
      storage:
        data:
          mount: /data
          readOnly: false
```

> Notes:
> - **Max 2 volumes per profile** on Akash — one ephemeral root + one persistent
>   `data` is exactly the limit. Don't add a third.
>   <https://medium.com/@mancheaster59/persistent-storage-on-akash-and-a-deployment-guide-for-how-to-deploy-it-c8f0127cd884>
> - Ephemeral dropped 200Gi → 30Gi on purpose: keeping your footprint small
>   makes you the *last* pod evicted under node disk pressure.
> - `beta3`=NVMe, `beta2`=SSD, `beta1`=HDD.
>   <https://akash.network/docs/providers/getting-started/hardware-requirements/>

### 1b. `akash/install.sh` → install into the persistent venv, skip if present

```bash
#!/usr/bin/env bash
set -euo pipefail
VENV="${VENV:-/data/venv}"
STATE_DIR="${STATE_DIR:-/data/state}"
mkdir -p "$STATE_DIR"

# Idempotent: if torch already imports from the persistent venv, skip the whole install.
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch, flash_attn" 2>/dev/null; then
  echo "[install] venv already populated — skipping (saved ~150s + download)"
  touch "$STATE_DIR/INSTALL_OK"
  exit 0
fi

python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

# --- your existing pinned installs, but into the venv (PIP_CACHE_DIR is persistent) ---
"$VENV/bin/pip" install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
# flash_attn: keep YOUR existing prebuilt-wheel install command, just call "$VENV/bin/pip"
# "$VENV/bin/pip" install <your flash_attn 2.7.4.post1 cxx11abiFALSE wheel url>
"$VENV/bin/pip" install "transformer_lens==2.18.0" "nnsight==0.7.0" \
                        "transformers>=4.47.0" "accelerate>=0.34.0" \
                        huggingface_hub hf_transfer

"$VENV/bin/python" -c "import torch, flash_attn; print('install verified')"
touch "$STATE_DIR/INSTALL_OK"
echo "INSTALL_OK"
```

> Because the venv lives on `/data`, the ~150s install + ~800 MB torch download
> happen **once for the whole lease**, not once per eviction.

### 1c. `akash/predownload_models.py` → cache on `/data`, per-model markers, resume

```python
import os, pathlib
from huggingface_hub import snapshot_download

STATE = pathlib.Path(os.environ.get("STATE_DIR", "/data/state"))
STATE.mkdir(parents=True, exist_ok=True)
HF_HOME = os.environ.get("HF_HOME", "/data/hf_cache")   # set in SDL env

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-4-mini-instruct",
]

def slug(repo): return repo.replace("/", "__")

for repo in MODELS:
    marker = STATE / f"MODEL_{slug(repo)}_OK"
    if marker.exists():
        print(f"[predownload] {repo} already complete — skipping")
        continue
    print(f"[predownload] {repo} -> {HF_HOME}")
    # snapshot_download RESUMES partial downloads automatically (.incomplete blobs).
    # An eviction mid-download is recovered on the next boot.
    snapshot_download(
        repo_id=repo,
        cache_dir=HF_HOME,
        max_workers=4,            # moderate concurrency; lower to 2 if you still see pressure
        # token=os.environ.get("HF_TOKEN"),  # needed for gated repos (LLaMA/Gemma)
    )
    marker.touch()
    print(f"PREDOWNLOAD_OK {repo}")

(STATE / "PREDOWNLOAD_OK").touch()
print("PREDOWNLOAD_OK")
```

> **Gated models:** `Llama-3.1` and `gemma-2` require an HF token + accepted
> license. Pass `HF_TOKEN` as an SDL env var and uncomment the `token=` line, or
> you'll get 401s that look like failures.

### 1d. `akash/_full_pipeline.py` → checkpoint-driven, skips done stages

Make the orchestrator idempotent. Each stage checks its marker on `/data/state`
and is skipped if already done. Run everything through the persistent venv's
Python.

```python
import os, subprocess, pathlib, sys

STATE = pathlib.Path(os.environ.get("STATE_DIR", "/data/state"))
REPO  = pathlib.Path(os.environ.get("REPO_DIR", "/data/Audit_Benchmark"))
PY    = os.environ.get("VENV", "/data/venv") + "/bin/python"
STATE.mkdir(parents=True, exist_ok=True)

def done(name): return (STATE / name).exists()
def mark(name): (STATE / name).touch()

def step(marker, argv, cwd=REPO):
    if done(marker):
        print(f"[pipeline] {marker} already done — skip"); return
    print(f"[pipeline] running {marker}: {' '.join(map(str, argv))}", flush=True)
    subprocess.run(argv, cwd=str(cwd), check=True)
    mark(marker)

# 1) install (idempotent inside install.sh too)
step("INSTALL_OK", ["bash", str(REPO / "akash/install.sh")])
# 2) predownload all models (per-model markers handled inside the script)
step("PREDOWNLOAD_OK", [PY, str(REPO / "akash/predownload_models.py")])
# 3) GPU dry run (loads from /data cache; no network)
step("DRYRUN_OK", [PY, str(REPO / "Code/mirage/Dry_Run/dry_run_gpu_cpu.py"),
                   "--n-seeds", "2"])

print("PIPELINE_COMPLETE", flush=True)
mark("PIPELINE_COMPLETE")
```

### 1e. `akash/supervise_pipeline.sh` → re-launches on crash; resumes on reboot

```bash
#!/usr/bin/env bash
# Runs the resumable pipeline. On crash, retries. On pod reboot, the SDL command
# re-runs this and the pipeline resumes from /data/state markers.
set -uo pipefail
REPO="${REPO_DIR:-/data/Audit_Benchmark}"
PY="${VENV:-/data/venv}/bin/python"
STATE="${STATE_DIR:-/data/state}"
mkdir -p /data/logs

if [ -f "$STATE/PIPELINE_COMPLETE" ]; then
  echo "[supervise] pipeline already complete; nothing to do."
  exit 0
fi

ATTEMPT=0
until [ -f "$STATE/PIPELINE_COMPLETE" ]; do
  ATTEMPT=$((ATTEMPT+1))
  ts=$(date -u +%FT%TZ)
  echo "[supervise] attempt $ATTEMPT at $ts (pid1_age=$(ps -p 1 -o etimes=)s)"
  # python3 fallback if venv not built yet on a very fresh boot
  RUNPY="$PY"; [ -x "$PY" ] || RUNPY="python3"
  "$RUNPY" "$REPO/_full_pipeline.py" 2>&1 | tee -a "/data/logs/pipeline.${ts}.log"
  if [ -f "$STATE/PIPELINE_COMPLETE" ]; then break; fi
  echo "[supervise] pipeline exited without completion; resuming in 15s"
  sleep 15
done
echo "[supervise] PIPELINE_COMPLETE — supervisor exiting."
```

> Note: your entrypoint was `python akash/_full_pipeline.py` from repo root.
> Keep that for manual runs; the supervisor just wraps it with retry + the
> persistent venv. Both paths are now idempotent.

**Net effect:** an eviction now costs ~30s (re-apt + re-clone-pull + resume),
not ~150s+42 GB. Within a few eviction cycles the whole run finishes even if no
single container lives past 10 minutes.

---

## P2 — Shrink the disk footprint (removes the trigger)

1. **Use the runtime image, not devel.** `nvidia/cuda:12.4.1-cudnn-devel-...`
   is ~8–9 GB; the **runtime** variant is far smaller. You install a *prebuilt*
   `flash_attn` wheel, so you don't need `nvcc`:
   ```yaml
   image: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
   ```
   Test this in a throwaway deploy first — if the prebuilt wheel imports, you've
   cut several GB off `imagefs`. (Roll back to `devel` only if something needs
   the toolkit.)
2. **All caches already point to `/data`** via the SDL env (`HF_HOME`,
   `PIP_CACHE_DIR`, `XDG_CACHE_HOME`). Confirm nothing in your code hardcodes
   `~/.cache` or `/workspace` for model weights.
3. **Clean apt lists** in the command (`rm -rf /var/lib/apt/lists/*`) to shave
   ephemeral bytes if you stay on an apt-in-command approach.
4. **Lower `max_workers`** in `snapshot_download` (4 → 2) only if the watchdog
   still shows disk pressure on `/data` (unlikely, since `/data` is off the
   contended ephemeral disk).

---

## P3 — Provider + GPU right-sizing (get off the bad host)

### 3a. You almost certainly don't need an A100-80GB

Your dry run loads models **sequentially** (peak VRAM ≈ one model ≈ **16 GB**
for LLaMA-8B). An A100-80GB is overkill, scarce, expensive, and lands you on the
most over-subscribed hosts. A **24 GB** card is enough for sequential loading;
**48 GB** if your *full* pipeline ever needs two models co-resident (8B+7B ≈
30 GB).

> ⚠️ **Assumption to verify:** that your real workload can run sequentially like
> the dry run. If it must hold all 4 models at once (~42 GB), you need ~48 GB
> VRAM and should keep targeting larger cards.

Request a set of widely-available GPUs by model (the `model:` attribute is
supported; values like these appear in real Akash GPU SDLs):

```yaml
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
                - model: rtx4090     # 24 GB
                - model: rtx3090     # 24 GB
                - model: a10         # 24 GB
                - model: l4          # 24 GB
                - model: a40         # 48 GB
                - model: a6000       # 48 GB
                - model: l40         # 48 GB
                - model: l40s        # 48 GB
```

> I'm confident `model:` filtering works (it's in published Akash GPU SDLs). I'm
> **not** certain Akash currently exposes a GPU-`ram`/VRAM attribute to filter by
> memory size directly — verify that against
> <https://akash.network/docs/> before relying on a `ram:` field, rather than
> assuming it exists.
> GPU model list: <https://akash.network/pricing/gpus/>

Widening the GPU set + smaller card = **far more bids**, **lower $/hr**, and a
good chance of a **less-contended provider** than the A100 host with load 14.

### 3b. Pick a calmer provider

There is **no SDL field to query host load before bidding**, so use one of:

- **Probe before committing.** Deploy a *cheap* container (the watchdog only,
  minimal/no GPU) to a candidate provider for ~3–5 minutes, read
  `cat /proc/loadavg` and `df -h`, then deploy the real workload to the calmest
  one. A few cents of probing beats repeated failed A100 runs.
- **Require audited providers** via `signedBy` in the `placement` section.
- **Use Akash Console** to filter by region/uptime and to see which providers
  actually offer persistent `beta3`/`beta2` + your GPU.
- If `beta3` (NVMe) gets **no bids**, switch the volume to `class: beta2` (SSD) —
  broader availability, still fine for HF cache I/O.

---

## P4 — (Optional) Prebuilt Docker image = zero install time

The most robust long-term setup: build an image `FROM
nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` with apt deps + the full pinned
Python stack (torch, flash_attn wheel, transformer_lens, nnsight, transformers,
accelerate) **baked in**, push to Docker Hub / GHCR, and set it as the SDL
`image`. Then container boot has **no install phase at all** — only model
downloads remain, and those already go to `/data`.

- Trade-off: a ~12–15 GB image takes a one-time pull and uses `imagefs`. Keep
  models *out* of the image (they belong on the persistent volume).
- This shrinks the dangerous window to almost nothing and is worth doing once
  the pipeline is stable.

---

## Answers to your 8 questions

1. **Guaranteed QoS to stop eviction?** Akash sets pod *limits = your SDL
   request* (your `memory.max` = 64 GiB confirms it), so for CPU/memory your pod
   is most likely **already Guaranteed**. But it won't help here: **QoS
   explicitly excludes `ephemeral-storage`**, so a Guaranteed pod is still
   evicted on disk pressure or an ephemeral-limit breach. QoS is not your lever.
2. **Is `beta3` persistent storage reliable / how many providers?** It works and
   **persists for the lease's lifetime** (lost only on lease close, deployment
   update, or migration — not on a restart). Fewer providers offer it, *fewer
   still* with an A100. No clean SDL filter — use Akash Console, and fall back to
   `beta2` (SSD) for more bids.
3. **A "dedicated"/bare-metal flag?** No universal protocol flag like
   `dedicated: true`. Some providers offer bare-metal/single-tenant GPU; identify
   them via provider attributes, Akash Console, or community lists, and require
   audited providers with `signedBy`.
4. **node-affinity / pod priority in SDL?** Not exposed to tenants. You influence
   *placement* via provider attributes + pricing, not via Kubernetes
   `nodeAffinity`/`priorityClass`. Don't spend time looking for these fields.
5. **Is load 14 expected / how to find low-load providers?** It's a heavily
   shared node — common on popular A100 providers, not a guarantee everywhere.
   No pre-bid load query exists; use the **probe-then-deploy** approach in P3b
   plus a smaller, less-contended GPU type.
6. **`emptyDir: {medium: Memory}` / tmpfs?** Akash doesn't expose arbitrary
   Kubernetes volume types. There **is** a `class: ram` for shared memory (SHM),
   but it counts against your **memory** limit, can't realistically hold 42 GB,
   and doesn't persist — not a fix here. Use the persistent volume.
   (SHM/`ram` class: Akash SDL docs.)
7. **Known workaround for the ~10-min pattern?** Yes — what this doc describes:
   the pattern is disk/ephemeral (or node-pressure) eviction *during downloads*.
   Move heavy writes to a persistent volume, make the pipeline resumable, shrink
   the ephemeral footprint, and pick a calmer/right-sized provider.
8. **Refund for lost lease time?** Akash bills **per active lease block via
   escrow**; there's no protocol-level SLA refund for a workload that kept dying.
   Closing the lease returns **unspent** escrow but not consumed time. Some
   providers may offer goodwill off-protocol — ask the operator directly. Don't
   count on it.

---

## Anti-patterns — do NOT spend time on these

- ❌ **Bumping memory to 128Gi.** You use 253 MB. Memory is not the constraint,
  and bigger requests just shrink your bid pool. (Revisit only if the watchdog
  proves a memory spike.)
- ❌ **Relying on Guaranteed QoS** to prevent eviction — it doesn't cover disk.
- ❌ **A 42 GB memory-backed tmpfs** for the model cache — needs ~42 GB+ RAM and
  recreates the very pressure you're avoiding; doesn't persist.
- ❌ **Re-deploying to the same A100 host** (`provider.a100.dsm.val.akash.pub`)
  expecting a different result.
- ❌ **Keeping models/venv on ephemeral storage** "to get more bids." The whole
  fix is moving them off ephemeral.

---

## Cursor implementation checklist

- [ ] **`akash/watchdog.sh`** — create (P0 script). Logs to `/data/logs`.
- [ ] **`akash/supervise_pipeline.sh`** — create (P1e script).
- [ ] **`akash/install.sh`** — install into `$VENV` (`/data/venv`); idempotent
      skip if `torch`+`flash_attn` already import; keep pinned versions + your
      existing flash_attn wheel command, just via `"$VENV/bin/pip"`.
- [ ] **`akash/predownload_models.py`** — cache to `$HF_HOME` (`/data/hf_cache`),
      per-model markers in `$STATE_DIR`, rely on `snapshot_download` resume,
      handle `HF_TOKEN` for gated repos.
- [ ] **`akash/_full_pipeline.py`** — checkpoint-driven (`INSTALL_OK`,
      `PREDOWNLOAD_OK`, `DRYRUN_OK`, `PIPELINE_COMPLETE`); run via `$VENV` python.
- [ ] **`Code/mirage/Dry_Run/dry_run_gpu_cpu.py`** — load models from
      `$HF_HOME`; ensure a clean exit so `DRYRUN_OK` is written; keep sequential
      load/unload.
- [ ] **`akash/_deploy_mirage.py`** — emit the P1a SDL: persistent `data` volume
      + `params.storage.data.mount: /data`, 30Gi ephemeral, all cache env vars,
      supervised command, widened/right-sized GPU set (P3a). Add a `class: beta2`
      fallback toggle and an optional `HF_TOKEN` env.
- [ ] **`config.py`** — document GPU sizing (sequential ⇒ 24 GB OK; co-resident
      ⇒ 48 GB).
- [ ] **(P2)** Try `...-runtime-...` image in a throwaway deploy; keep `devel`
      only if the prebuilt wheel fails to import.
- [ ] **(P4, later)** Prebuilt image with the full stack baked in.

### Acceptance test

1. Deploy with the new SDL to a **non-A100, calmer** provider (probe first).
2. SSH in, confirm `mount | grep /data` shows the persistent volume and
   `echo $HF_HOME` = `/data/hf_cache`.
3. Let the supervisor run. **Force-kill** the pod once (or just wait for an
   eviction) and confirm on reboot it **resumes** (markers in `/data/state`
   present; no re-install, no full re-download).
4. After completion, `cat /data/state/PIPELINE_COMPLETE` exists and
   `tail /data/logs/watchdog.log` shows what (if anything) was spiking — closing
   the loop on root cause.