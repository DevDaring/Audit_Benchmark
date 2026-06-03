#!/usr/bin/env bash
# Dry-run launcher — writes ZERO files to results/
# Logs to ~/dryrun.log
set -euo pipefail

MIRAGE=/home/koushikdeb2009/Audit_Benchmark/Code/mirage
# Models are stored directly in hf_cache/ (not hf_cache/hub/) so use HF_HUB_CACHE
export HF_HUB_CACHE=/home/koushikdeb2009/hf_cache
export HF_HOME=/home/koushikdeb2009/hf_cache
export MIRAGE_SEQUENTIAL_MODELS=1
export MIRAGE_EVAL_BATCH_SIZE=4

cd "$MIRAGE"
set -a; source .env; set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN}"

echo "[$(date -u +%FT%TZ)] === DRY RUN START ===" | tee ~/dryrun.log
python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2 2>&1 | tee -a ~/dryrun.log
EXIT=$?
echo "[$(date -u +%FT%TZ)] === DRY RUN EXIT:$EXIT ===" | tee -a ~/dryrun.log
exit $EXIT
