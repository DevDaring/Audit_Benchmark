# MIRAGE — Akash GPU Deployment

## Quick Start

### 1. Generate the SDL and get deployment instructions

```bash
python akash/deploy.py
```

This writes `akash/deployment.yaml` (the Akash SDL) and prints step-by-step
Console UI instructions.  Your `.env` file is base64-encoded inside the SDL
so it is injected into the container at boot — it never touches Git.

### 2. Deploy via Akash Console (recommended for first-timers)

1. Go to <https://console.akash.network/>
2. Connect your Keplr/Leap wallet (needs ~2 AKT deposit)
3. Click **Deploy → From SDL file**
4. Upload `akash/deployment.yaml`
5. Wait for bids, select cheapest A100 40 GB provider
6. Accept lease → note the SSH IP and forwarded port

### 3. Verify and install

```bash
# Upload .env, run install.sh, run dry_run_all.py
python akash/run_install.py --host <IP> --port <PORT>
```

Or step-by-step:

```bash
python akash/run_install.py --host <IP> --port <PORT> --action upload_env
python akash/run_install.py --host <IP> --port <PORT> --action install
python akash/run_install.py --host <IP> --port <PORT> --action dry_run
```

### 4. Full health check

```bash
python akash/check_vm.py --host <IP> --port <PORT>
# add --full for the GPU dry run
```

### 5. Launch GPU pipeline

```bash
python akash/run_install.py --host <IP> --port <PORT> --action gpu_run
```

Then SSH in and attach to the tmux session to watch progress:

```bash
ssh root@<IP> -p <PORT>
tmux attach -t gpu_run
tail -f /workspace/logs/gpu_run.log
```

---

## Flash-Attention Installation Notes

`install.sh` uses a layered strategy:

| Priority | Method | Notes |
|---|---|---|
| 1 | Direct official prebuilt wheel | `flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` — ~183 MB, no compilation, ~450K downloads |
| 2 | `pip install flash-attn --no-build-isolation` | Auto-downloads matching wheel or compiles from source (30-60 min) |
| 3 | Fail with clear error message | Check Python / CUDA / torch version match |

**Why `cxx11abiFALSE`**: pip-installed PyTorch always uses the pre-CXX11 ABI for broad Linux compatibility. flash-attn must match.

**Why PyTorch 2.6**: TransformerLens 2.18.0 is compatible with torch 2.6, and the official flash-attn 2.7.4.post1 prebuilt wheel for torch 2.6 has 1.3M downloads — the most battle-tested combination.

---

## VRAM Budget (A100 40 GB)

| Stage | Models in VRAM | Usage |
|---|---|---|
| Behavioral (Llama 8B) | HF model | ~16 GB |
| Behavioral (Gemma 9B) | HF model | ~18 GB |
| Behavioral (Qwen 7B) | HF model | ~14 GB |
| Behavioral (Phi-4 14B) | HF model | ~28 GB |
| CDVA (Llama 8B) | HF + TL (freed if tight) | ~16-32 GB |
| CDVA (Gemma 9B) | HF + TL (freed if tight) | ~18-36 GB |

For TL models (Llama, Gemma), `cdva_patching.py` automatically frees the HF model before TransformerLens conversion if free VRAM < 1.5× model size, then reloads the HF model afterwards.

---

## Files

| File | Purpose |
|---|---|
| `deploy.py` | Generate SDL + submit via CLI or print Console instructions |
| `deployment.yaml` | Generated SDL (git-ignored; contains base64 .env) |
| `install.sh` | Package installer (PyTorch, flash-attn, TL, nnsight, …) |
| `startup.sh` | Container boot script (clone repo, write .env, start SSH) |
| `run_install.py` | Remote orchestrator (upload_env / install / dry_run / gpu_run) |
| `check_vm.py` | Full VM health check via SSH |
