#!/usr/bin/env bash
# On-VM autonomous guard — finishes regen, validates pentad, starts supervisor.
# Survives local SSH disconnects. Safe to run alongside regen.
set -uo pipefail

REPO="${REPO_DIR:-/data/Audit_Benchmark}"
STATE="${STATE_DIR:-/data/state}"
MIRAGE="$REPO/Code/mirage"
PY="${VENV:-/data/venv}/bin/python"
LOG=/data/logs/autonomous_guard.log
LOCK=/data/state/autonomous_guard.lock

mkdir -p /data/logs "$STATE"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

if [ -f "$STATE/PIPELINE_COMPLETE" ]; then
  exit 0
fi

# Single instance
exec 9>"$LOCK"
if ! flock -n 9; then
  exit 0
fi

# Never fight an active regen
if pgrep -f 'regenerate_api_slots.py' >/dev/null 2>&1; then
  exit 0
fi

# Resume dead regen if checkpoints exist and pentad incomplete
"$PY" - <<'PY' >/tmp/guard_status.txt 2>&1 || true
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
from pathlib import Path
p = Path("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
df = pd.read_parquet(p)
de = int(df["slot"].isin(["d", "e"]).sum())
need = df[df.seed_source.str.lower().isin(["bbq","crows_pairs","stereoset"])].seed_id.nunique() * 5
print("de", de, "need", need)
try:
    from Dataset.validate_pentad import assert_production_ready
    assert_production_ready(df)
    print("ready", "yes")
except Exception as e:
    print("ready", "no")
    print("err", str(e)[:200])
PY

READY=no
DE=0
NEED=2980
grep -q '^ready yes' /tmp/guard_status.txt && READY=yes
DE=$(grep '^de ' /tmp/guard_status.txt | awk '{print $2}')
NEED=$(grep '^de ' /tmp/guard_status.txt | awk '{print $4}')

CTX="$MIRAGE/Dataset/seeds/context_shift_checkpoint.json"
COT="$MIRAGE/Dataset/seeds/cot_attack_checkpoint.json"

if [ "$READY" != "yes" ]; then
  if { [ -f "$CTX" ] || [ -f "$COT" ]; } && [ "${DE:-0}" -lt "${NEED:-2980}" ]; then
    log "regen died with checkpoint — restarting with --keep-checkpoint"
    cd "$MIRAGE"
    nohup "$PY" regenerate_api_slots.py --keep-checkpoint >> /data/logs/prelaunch_regen.log 2>&1 &
    exit 0
  fi
  if [ "${DE:-0}" -lt "${NEED:-2980}" ]; then
    log "pentad missing d/e — starting regen"
    cd "$MIRAGE"
    nohup "$PY" regenerate_api_slots.py >> /data/logs/prelaunch_regen.log 2>&1 &
  fi
  exit 0
fi

# Production-ready pentad — ensure DATASET_OK and supervisor
log "pentad production-ready — ensuring DATASET_OK + supervisor"

rm -f "$STATE/GPU_PIPELINE_OK" "$STATE/PIPELINE_COMPLETE"
rm -f "$MIRAGE/results/behavioral_results.parquet" "$MIRAGE/results/cdva_results.parquet" "$MIRAGE/results/tau_calibration.json"
rm -f "$CTX" "$COT"

touch "$STATE/DATASET_OK"

if ! pgrep -f 'supervise_pipeline.sh' >/dev/null 2>&1; then
  log "starting supervise_pipeline.sh"
  nohup bash "$REPO/akash/supervise_pipeline.sh" >> /data/logs/supervise.log 2>&1 &
else
  log "supervisor already running"
fi
