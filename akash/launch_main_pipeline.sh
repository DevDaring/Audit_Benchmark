#!/usr/bin/env bash
# Main GPU pipeline launcher.
# SAFETY: writes to Code/mirage/results/ only.
#         Dry run and test data never touch this directory.
set -euo pipefail

MIRAGE=/home/koushikdeb2009/Audit_Benchmark/Code/mirage
LOG=~/mirage_prod.log
# Models are stored directly in hf_cache/ (not hf_cache/hub/) so use HF_HUB_CACHE
export HF_HUB_CACHE=/home/koushikdeb2009/hf_cache
export HF_HOME=/home/koushikdeb2009/hf_cache

cd "$MIRAGE"
set -a; source .env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN}"
# Re-export after .env so secrets file cannot disable sequential mode on 40GB GPU.
export MIRAGE_SEQUENTIAL_MODELS=1
export MIRAGE_EVAL_BATCH_SIZE=4
export STATE_DIR="${HOME}/mirage-state"

echo "[$(date -u +%FT%TZ)] === PRODUCTION PIPELINE START ===" | tee "$LOG"
echo "Sequential loading: MIRAGE_SEQUENTIAL_MODELS=$MIRAGE_SEQUENTIAL_MODELS" | tee -a "$LOG"
echo "Batch size:         MIRAGE_EVAL_BATCH_SIZE=$MIRAGE_EVAL_BATCH_SIZE" | tee -a "$LOG"
echo "Results dir:        $MIRAGE/results/" | tee -a "$LOG"
echo "" | tee -a "$LOG"

python3 GPU_CPU/run_gpu_pipeline.py 2>&1 | tee -a "$LOG"
EXIT=$?

echo "" | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === PRODUCTION PIPELINE EXIT:$EXIT ===" | tee -a "$LOG"
exit $EXIT
