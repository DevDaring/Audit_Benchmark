#!/usr/bin/env bash
# bootstrap.sh -- VM entrypoint for the GPU_Remaining run on Akash (H200/H100).
# Secrets (HUGGINGFACE_TOKEN, Github_Classic_Token) arrive as container env vars
# injected by the SDL; they are never committed to the repo.
#
# Steps: system deps -> python deps -> precompiled flash-attn -> download models
#        (dataset is already in the cloned repo) -> DRY RUN (2 instances) ->
#        on pass, clean test artifacts -> MAIN run under a restart supervisor.
set -uo pipefail

WORK=/workspace
REPO="$WORK/Audit_Benchmark"
GR="$REPO/Code/mirage/GPU_Remaining"
export HF_HOME="$WORK/hf"
export DEBIAN_FRONTEND=noninteractive
export PIP="pip3 install --break-system-packages"

echo "[bootstrap] system deps"
apt-get update -y
apt-get install -y --no-install-recommends git wget ca-certificates python3 python3-pip python3-venv build-essential

echo "[bootstrap] python: $(python3 --version)"   # expect 3.12 on Ubuntu 24.04
cd "$GR"

echo "[bootstrap] write .env from injected secrets (gitignored, local only)"
python3 - <<'PY'
import os
keys = ["HUGGINGFACE_TOKEN", "Github_Classic_Token", "RANDOM_SEED"]
with open(".env", "w") as f:
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            f.write(f"{k}={v}\n")
print("wrote .env with", sum(1 for k in keys if os.environ.get(k)), "secrets")
PY

echo "[bootstrap] torch 2.5.1 (cu124)"
$PIP --upgrade pip
$PIP torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
echo "[bootstrap] python deps"
$PIP -r requirements_gpu.txt

echo "[bootstrap] precompiled flash-attention"
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget -q "$FA_URL" -O /tmp/fa.whl && $PIP --no-deps /tmp/fa.whl \
  && echo "[bootstrap] flash-attn installed" \
  || echo "[bootstrap] WARN flash-attn install failed; will fall back to sdpa/eager"

echo "[bootstrap] verify GPU"
python3 -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "[bootstrap] download OSM models (dataset already in repo)"
python3 - <<'PY'
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from config import OSM_MODELS, HUGGINGFACE_TOKEN
from huggingface_hub import snapshot_download
for m in OSM_MODELS:
    print("downloading", m["hf_id"], flush=True)
    snapshot_download(m["hf_id"], token=HUGGINGFACE_TOKEN)
print("models present")
PY

echo "[bootstrap] DRY RUN (2 instances)"
if python3 run_gpu_remaining.py --mode dry; then
    echo "[bootstrap] DRY PASSED -- removing test results+logs"
    rm -rf results/dryrun
    rm -f logs/*.log
else
    echo "[bootstrap] DRY FAILED -- aborting main; keeping container alive for inspection"
    sleep infinity
fi

echo "[bootstrap] MAIN run (restart supervisor)"
ATTEMPT=0
while true; do
    ATTEMPT=$((ATTEMPT+1))
    echo "[bootstrap] main attempt $ATTEMPT"
    python3 run_gpu_remaining.py --mode main && break
    echo "[bootstrap] main exited non-zero; retry in 60s"
    sleep 60
done

echo "[bootstrap] COMPLETE"
sleep infinity
