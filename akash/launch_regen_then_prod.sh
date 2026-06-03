#!/usr/bin/env bash
# Regenerate d/e API slots, validate pentad, then start production GPU pipeline.
set -euo pipefail
LOG=~/mirage_full_run.log

echo "[$(date -u +%FT%TZ)] === REGEN d/e THEN PRODUCTION ===" | tee "$LOG"

if ! ~/launch_regen_api_slots.sh 2>&1 | tee -a "$LOG"; then
  echo "[$(date -u +%FT%TZ)] ABORT: API regen failed — production NOT started" | tee -a "$LOG"
  exit 1
fi

echo "[$(date -u +%FT%TZ)] API regen OK — starting production pipeline" | tee -a "$LOG"
exec ~/launch_main_pipeline.sh 2>&1 | tee -a "$LOG"
