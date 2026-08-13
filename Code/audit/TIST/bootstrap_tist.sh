#!/usr/bin/env bash
# bootstrap_tist.sh -- Akash entrypoint for the TIST resubmission GPU battery.
#
# Secrets (HUGGINGFACE_TOKEN, Github_Classic_Token) arrive as container env vars
# injected by the SDL and are never committed.
#
# Steps: system deps -> torch cu124 -> python deps -> transformer_lens (--no-deps)
#        -> PRECOMPILED flash-attention wheel -> hf_transfer -> model download
#        -> DRY RUN on the real code path -> MAIN run under a restart supervisor,
#        with a background pusher syncing results to GitHub every 15 minutes.
#
# Speed notes (why these choices):
#   * flash-attn ships as a precompiled wheel matched to cu12 / torch 2.5 / cp312.
#     Building it from source on a leased GPU costs one to two hours of paid time.
#     GPU_CPU/load_osm.py already requests attn_implementation="flash_attention_2",
#     so installing the wheel is all that is needed; absent it, transformers falls
#     back to sdpa, which is numerically equivalent but slower.
#   * hf_transfer replaces the default HTTP client for hub downloads. The four
#     models are about 45 GB together and download time is billed like compute.
#   * MIRAGE_EVAL_BATCH_SIZE batches the behavioural generation; the patching
#     tasks are inherently sequential because each pair needs its own cache.
set -uo pipefail

WORK=/workspace
REPO="$WORK/Audit_Benchmark"
AUDIT="$REPO/Code/audit"
TIST="$AUDIT/TIST"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive
export PIP="pip3 install --break-system-packages"
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export PYTHONPATH="$AUDIT:${PYTHONPATH:-}"

echo "[tist] system deps"
apt-get update -y
apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-pip python3-venv build-essential

echo "[tist] python: $(python3 --version)"
cd "$AUDIT"

echo "[tist] write .env from injected secrets (gitignored)"
python3 - <<'PY'
import os
real = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED"]
# config.py _require()s these at import. The GPU battery never calls an external
# API, so non-empty dummies satisfy validation without shipping real keys here.
dummy = ["DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2",
         "OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2",
         "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4",
         "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "MISTRAL_API_KEY1", "MISTRAL_API_KEY2"]
n = 0
with open(".env", "w") as f:
    for k in real:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n"); n += 1
    for k in dummy:
        f.write(f"{k}=unused-by-tist-gpu-job\n")
print(f"wrote .env with {n} real secrets")
PY

echo "[tist] configure git"
git config --global --add safe.directory "$REPO"
git -C "$REPO" config user.name "MIRAGE TIST Runner"
git -C "$REPO" config user.email "koushikdeb2009@gmail.com"
git -C "$REPO" config pull.rebase true
if [ -n "${Github_Classic_Token:-}" ]; then
  git -C "$REPO" remote set-url origin "https://${Github_Classic_Token}@github.com/DevDaring/Audit_Benchmark.git"
fi

push_results() {
  mkdir -p "$AUDIT/results/tist" "$TIST/logs"
  echo "$1 @ $(date -u)" > "$AUDIT/results/tist/BOOT_STATUS.txt"
  git -C "$REPO" add -f Code/audit/results/tist Code/audit/TIST/logs >/dev/null 2>&1
  git -C "$REPO" commit -q -m "tist-gpu: $1" >/dev/null 2>&1
  git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1
  git -C "$REPO" push -q origin main >/dev/null 2>&1 && echo "[tist] pushed: $1"
}

push_results "container started ($(nvidia-smi -L 2>/dev/null | head -1))"

echo "[tist] torch 2.5.1 (cu124)"
$PIP --upgrade pip
$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

echo "[tist] python deps"
$PIP -r "$AUDIT/GPU_Remaining/requirements_gpu.txt"
$PIP scikit-learn hf_transfer

echo "[tist] transformer_lens 2.18.0 (--no-deps: keeps torch 2.5.1 / transformers 4.50.3)"
$PIP --no-deps transformer_lens==2.18.0

echo "[tist] precompiled flash-attention (cu12 / torch2.5 / cp312)"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
if wget -q "$FA_URL" -O /tmp/fa.whl && $PIP --no-deps /tmp/fa.whl; then
  echo "[tist] flash-attn installed"
else
  echo "[tist] WARN flash-attn wheel failed; transformers will use sdpa (slower, same numerics)"
fi

mkdir -p "$TIST/logs"
echo "[tist] verify stack"
python3 - > "$TIST/logs/verify.log" 2>&1 <<'PY'
import importlib.metadata as m
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
import transformer_lens, nnsight  # noqa: F401
print("transformer_lens", m.version("transformer_lens"), "| nnsight", m.version("nnsight"))
try:
    import flash_attn
    print("flash_attn", m.version("flash_attn"))
except Exception as e:
    print("flash_attn ABSENT:", e)
PY
VRC=$?
cat "$TIST/logs/verify.log"
if [ "$VRC" -ne 0 ]; then
  echo "[tist] FATAL: stack verification failed; container kept alive"
  push_results "FATAL stack verify (see TIST/logs/verify.log)"
  sleep infinity
fi

echo "[tist] download models"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath("."))
from config import OSM_MODELS, HUGGINGFACE_TOKEN
from huggingface_hub import snapshot_download
for m in OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=HUGGINGFACE_TOKEN)
print("models present")
PY

push_results "setup complete; starting dry run"

echo "[tist] DRY RUN (2 units per task, real code path)"
cd "$AUDIT"
python3 TIST/run_tist_gpu.py --tasks all --dry > "$TIST/logs/dryrun.log" 2>&1; DRY_RC=$?
tail -60 "$TIST/logs/dryrun.log"
push_results "dry run rc=$DRY_RC"
if [ "$DRY_RC" -ne 0 ]; then
  echo "[tist] DRY FAILED; container kept alive for inspection"
  sleep infinity
fi

echo "[tist] dry passed; clearing dry-run records so they never enter the real results"
rm -f "$AUDIT"/results/tist/e1/*.jsonl "$AUDIT"/results/tist/e4/*.jsonl
rm -f "$AUDIT"/results/tist/e4/behav_*.parquet

echo "[tist] background pusher every 15 min"
( while true; do sleep 900; push_results "checkpoint"; done ) &
PUSHER=$!

echo "[tist] MAIN run"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[tist] main attempt $ATTEMPT"
  # Pull before each attempt. This picks up the Hindi and Bengali pentads if they were
  # committed after the lease started, and lets a code fix reach a live lease without
  # paying to reprovision. Results are force-added, so reset them before pulling to
  # avoid a rebase conflict against the runner's own pushes.
  git -C "$REPO" stash -q --include-untracked >/dev/null 2>&1 || true
  git -C "$REPO" pull --rebase -q origin main >/dev/null 2>&1 || true
  git -C "$REPO" stash pop -q >/dev/null 2>&1 || true
  python3 TIST/run_tist_gpu.py --tasks all > "$TIST/logs/main.log" 2>&1 && break
  tail -40 "$TIST/logs/main.log"
  push_results "main attempt $ATTEMPT exited non-zero"
  sleep 60
done

kill "$PUSHER" 2>/dev/null || true
touch "$AUDIT/results/tist/TIST_GPU_DONE"
push_results "COMPLETE"
echo "[tist] COMPLETE"
sleep infinity
