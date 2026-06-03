#!/usr/bin/env bash
# One-time: regenerate slot d/e to match patched slot-a (pre-GPU, not during experiment).
set -euo pipefail
MIRAGE=/home/koushikdeb2009/Audit_Benchmark/Code/mirage
LOG=~/regen_api_slots.log
cd "$MIRAGE"
set -a; source .env; set +a

echo "[$(date -u +%FT%TZ)] === REGENERATE API SLOTS d/e START ===" | tee "$LOG"
python3 regenerate_api_slots.py 2>&1 | tee -a "$LOG"
EXIT=$?
echo "[$(date -u +%FT%TZ)] === REGENERATE EXIT:$EXIT ===" | tee -a "$LOG"
exit $EXIT
