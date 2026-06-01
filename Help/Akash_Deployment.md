# PEAT — Akash A100 Deployment Guide

Complete reference for deploying and operating the PEAT research pipeline on Akash Network.
A coding agent can read this file and execute every step end-to-end.

---

## 1. Prerequisites

| Requirement | Detail |
|-------------|--------|
| Python (local) | Any 3.x with `pip` |
| `requests` | Auto-installed by `deploy.py` if missing |
| `paramiko` | Required for `run_install.py` / `check_vm.py` — `pip install paramiko` |
| `.env` file | Must exist at `Code/.env` with all keys filled in |
| GitHub access | Repo must be pushed — container clones from GitHub |

---

## 2. Key Credentials

All credentials live in `Code/.env`. **Never hardcode them elsewhere.**

| Key name in `.env` | Purpose |
|--------------------|---------|
| `Akash_API_Key` | Akash Console API — creates/manages deployments |
| `Github_Classic_Token` | Clones private repo inside container (`ghp_...`) |
| `HF_Classic_Token` | HuggingFace model downloads |
| `GCP_Key1`–`GCP_Key4` | Gemini 2.5 Flash Lite (LLM-as-judge, round-robin) |
| `Deepseek_API_key` | DeepSeek fallback check in dry run |
| `Mistral_API_Key` | Mistral fallback check in dry run |

Current live values are in `Code/.env` (not repeated here to avoid accidental exposure).

---

## 3. Scripts Reference

All scripts live in the `akash/` directory, run from the **repo root** (`D:\PhD\PEAT_Debias`):

| Script | Purpose |
|--------|---------|
| `akash/deploy.py` | Fresh deploy OR resume/close an existing deployment |
| `akash/run_install.py` | SSH in, `git pull`, re-run `install.sh` on existing VM |
| `akash/check_vm.py` | SSH in, poll until available, print install/GPU status |
| `akash/startup.sh` | Container boot script (not run locally — injected as base64) |
| `akash/last_dseq.txt` | Auto-saved after each deploy: dseq, provider, SSH address |
| `akash/last_manifest.txt` | Auto-saved manifest — required for `--resume` |

---

## 4. Deployment Lifecycle

### 4a. Fresh deploy (new VM from scratch)

```bash
# From repo root:
python akash/deploy.py
```

What happens internally:
1. Reads `Code/.env` → extracts `Akash_API_Key` + `Github_Classic_Token`
2. Base64-encodes `.env` and `akash/startup.sh` into the SDL (never written to disk)
3. `POST /v1/deployments` — creates on-chain deployment with **$20 escrow deposit**
4. Polls `GET /v1/bids?dseq=` every 5 s (up to 300 s) until GPU providers bid
5. Accepts the **cheapest bid** (`POST /v1/leases`)
6. Polls lease status every 10 s until SSH host:port are assigned
7. Prints SSH command and saves `akash/last_dseq.txt`

Container boot sequence (inside the VM, takes ~25–30 min total):
- `[0/7]` SSH daemon starts **first** (always reachable even if install fails)
- `[1/7]` System packages (`curl`, `git`, `tmux`, `ninja-build`, etc.)
- `[2/7]` Python pip (uses system Python 3.10 from CUDA image)
- `[3/7]` MOTD written
- `[4/7]` Repo cloned: `git clone https://<GIT_TOKEN>@github.com/DevDaring/PEAT_Debias.git`
- `[5/7]` `.env` decoded from `ENV_B64` env var and written to `/workspace/PEAT_Debias/Code/.env`
- `[6/7]` `bash install.sh` (PyTorch, PEFT, flash-attn wheel, all packages — ~25 min)
- `[7/7]` Dry run launched inside `tmux` session `peat`

**Example output:**
```
============================================================
 DEPLOYMENT LIVE
============================================================
  DSEQ    : 26628620
  Provider: akash12v6dhc8awlwhv438jjyw80eguhgtm735mfv3fx
  Cost    : ~$2.40/hour
  SSH     : ssh root@provider.a100.dsm.val.akash.pub -p 31133
  Password: peat2026!
============================================================
```

### 4b. Resume an existing deployment (already deployed, no active lease)

Use this if bids were received but the lease was not accepted, or after a transient API error:

```bash
python akash/deploy.py --resume <dseq>
```

- If `akash/last_manifest.txt` exists, uses it directly
- If a lease is already active, skips to SSH wait
- Otherwise waits for bids and creates a new lease

### 4c. Close deployment (stop billing)

```bash
python akash/deploy.py --close <dseq>
```

Calls `DELETE /v1/deployments/{dseq}`. Billing stops immediately. Container is destroyed.

### 4d. Inspect the SDL without deploying

```bash
python akash/deploy.py --show-sdl
```

Prints the YAML SDL with secrets redacted (`GIT_TOKEN_REDACTED`, `ENV_B64_REDACTED`).

---

## 5. Connecting to a Live VM

```bash
# Current live VM (as of last run):
ssh root@provider.a100.dsm.val.akash.pub -p 31133
# Password: peat2026!
```

After login:
```bash
tmux attach -t peat          # attach to dry-run / pipeline session
tail -f /workspace/dryrun.log    # watch dry-run output
tail -f /workspace/startup.log   # watch container boot log
tail -f /workspace/install.log   # watch package install log
```

---

## 6. After Container is Ready (post-install)

### Check install status from local machine

```bash
python akash/check_vm.py
```

Polls SSH until available, then prints startup + GPU status.

### Pull latest code changes + re-run install

After pushing local changes to GitHub:

```bash
python akash/run_install.py
```

This SSHes in, does `git pull origin main`, then re-runs `install.sh`.
Output is streamed live (~25 min for full install, much faster for code-only changes).

### Run the full pipeline (on VM)

```bash
# On the VM:
cd /workspace/PEAT_Debias/Code
python3 run_all.py
```

Or from a tmux session (recommended — survives disconnections):
```bash
tmux new-session -d -s run "cd /workspace/PEAT_Debias/Code && python3 run_all.py 2>&1 | tee /workspace/pipeline.log"
tmux attach -t run
```

### Re-run dry run only (on VM)

```bash
cd /workspace/PEAT_Debias/Code
python3 -c "
import sys, os; os.chdir('/workspace/PEAT_Debias/Code'); sys.path.insert(0,'.')
from peat.dryrun import run_dryrun
ok = run_dryrun(skip_if_recent=False)
print('PASSED' if ok else 'FAILED')
"
```

---

## 7. VM Specs (SDL constants in `deploy.py`)

| Resource | Value |
|----------|-------|
| GPU | NVIDIA A100 (40 GB or 80 GB — both accepted) |
| CPU | 12 vCPU |
| RAM | 64 GB |
| Storage | 250 GB |
| Container image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` |
| Python | 3.10.12 (system, from CUDA image) |
| Max bid price | 50 000 µUSDC/block ≈ $30/hr cap (actual ~$2–4/hr) |
| SSH password | `peat2026!` |

---

## 8. Workspace Layout on VM

```
/workspace/
  PEAT_Debias/
    Code/
      .env                     ← restored from base64 payload
      run_all.py               ← main pipeline entry point
      peat/                    ← core library
      state/run_state.json     ← resume checkpoint (pipeline progress)
      results/                 ← all outputs
      logs/                    ← all logs
    akash/                     ← deployment scripts
  startup.log                  ← container boot log
  install.log                  ← install.sh output
  dryrun.log                   ← dry-run output
```

---

## 9. Common Scenarios for a Coding Agent

### Scenario A: Deploy from scratch

1. Verify `Code/.env` has `Akash_API_Key` and `Github_Classic_Token`
2. Push all local code changes: `git push origin main`
3. Run: `python akash/deploy.py`
4. Wait for "DEPLOYMENT LIVE" output, note SSH address
5. SSH in after ~30 min, run `tmux attach -t peat` to check dry run
6. If dry run passes, run `python3 run_all.py` from `/workspace/PEAT_Debias/Code`

### Scenario B: VM exists, code was updated locally

1. Push changes: `git push origin main`
2. Run: `python akash/run_install.py`
3. This pulls and re-installs on the live VM

### Scenario C: Resume after interrupted pipeline

The pipeline is fully resumable. Simply re-run:
```bash
# On VM:
cd /workspace/PEAT_Debias/Code && python3 run_all.py
```

`run_all.py` reads `state/run_state.json` and skips all completed cells.

### Scenario D: Recover SSH address for existing deployment

```bash
cat akash/last_dseq.txt
# Shows: dseq=..., provider=..., ssh=...
```

Or check `https://console.akash.network/deployments/<dseq>` in a browser.

### Scenario E: Close when done

```bash
python akash/deploy.py --close $(grep dseq akash/last_dseq.txt | cut -d= -f2)
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `HTTP 403` from Akash API | Invalid/expired `Akash_API_Key` | Check key in `.env` |
| No bids after 300 s | Max bid too low or GPU region scarce | Increase `MAX_BID_UUSDC` in `deploy.py` (line ~26) |
| SSH connection refused | Container still booting | Wait; SSH starts within ~2 min of lease creation |
| `install.sh` errors | Package version conflict | Check `/workspace/install.log` on VM |
| `flash_attn` import fails | Wrong Python/CUDA version | Python on VM is 3.10 (not 3.12); the wheel in `install.sh` targets cp310+cu124 |
| `Gated model 403` | HuggingFace access not granted | Request access on HF for `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-4b-it`; ensure `HF_Classic_Token` matches the approved account |
| Pipeline OOM | VRAM exceeded | Reduce batch size in `peat/peat.py` (`batch_size` in `run_successive_halving`/`run_final_training`) |
| `state/dryrun_passed` missing | Dry run never ran / failed | Delete and re-run dry run (see §6) |
| Resume fails: empty manifest | `last_manifest.txt` deleted | Delete deployment and redeploy fresh |

---

## 11. Cost Estimate

| Phase | Duration | Cost (at $2.50/hr) |
|-------|----------|-------------------|
| Install + dry run | ~30 min | ~$1.25 |
| Stage 1 — PEAT core (4 models) | ~40 hr | ~$100 |
| Stage 2 — PEAT scaling (2 models) | ~20 hr | ~$50 |
| Stage 3 — Baselines (9 × 4 × 3) | ~80 hr | ~$200 |
| Stages 4–5 — Aggregation + figures | ~1 hr | ~$2.50 |
| **Total (approximate)** | **~141 hr** | **~$353** |

Close deployment immediately after pipeline completes to avoid idle charges.
