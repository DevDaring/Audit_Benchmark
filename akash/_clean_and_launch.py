"""
Clean VM state for a fresh production run while preserving:
  - /data/venv          (installed packages)
  - /data/hf_cache      (downloaded models)
  - /data/.env          (API keys, CRLF-normalised)
  - Dataset/seeds/*.parquet if already built
  - benchmark dataset caches under Code/mirage/cache/

Removes:
  - DRYRUN_OK, PIPELINE_COMPLETE, GPU_PIPELINE_OK state markers
  - Partial GPU results from any prior attempt
  - Stale pipeline logs (archived first)

Then: git pull + restart supervise_pipeline.sh
"""
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import paramiko

HOST, PORT, PW = "provider.a100.dsm.val.akash.pub", 31532, "MirageVM2026!"
REPO = "/data/Audit_Benchmark"
MIRAGE = f"{REPO}/Code/mirage"

CLEAN_SCRIPT = r"""#!/usr/bin/env bash
set -euo pipefail
STATE=/data/state
REPO=/data/Audit_Benchmark
MIRAGE=$REPO/Code/mirage
ARCHIVE=/data/logs/archive_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ARCHIVE"

echo "=== CLEAN VM (preserve models, venv, dataset) ==="

# Archive old pipeline logs
if ls /data/logs/pipeline_attempt_*.log 1>/dev/null 2>&1; then
  mv /data/logs/pipeline_attempt_*.log "$ARCHIVE/" 2>/dev/null || true
fi
[ -f /data/logs/supervise.log ] && cp /data/logs/supervise.log "$ARCHIVE/" || true

# Remove production state markers (keep INSTALL_OK, PREDOWNLOAD_OK, per-model markers)
for m in DRYRUN_OK DATASET_OK GPU_PIPELINE_OK PIPELINE_COMPLETE; do
  rm -f "$STATE/$m"
done

# Remove partial GPU results (will be regenerated; resume logic uses empty files)
rm -f "$MIRAGE/results/behavioral_results.parquet" \
      "$MIRAGE/results/cdva_results.parquet" \
      "$MIRAGE/results/tau_calibration.json" 2>/dev/null || true

# Normalise .env CRLF (Windows upload fix)
if [ -f /data/.env ]; then
  sed -i 's/\r$//' /data/.env
  cp /data/.env "$MIRAGE/.env"
  chmod 600 /data/.env "$MIRAGE/.env"
fi

# Stop any stale pipeline processes
pkill -f "dry_run_gpu_cpu.py" 2>/dev/null || true
pkill -f "run_gpu_pipeline.py" 2>/dev/null || true
pkill -f "_full_pipeline.py" 2>/dev/null || true
sleep 2

# Pull latest code
git -C "$REPO" pull --ff-only origin main

echo "=== PRESERVED ASSETS ==="
echo "venv:    $(du -sh /data/venv 2>/dev/null | cut -f1) at /data/venv"
echo "models:  $(du -sh /data/hf_cache 2>/dev/null | cut -f1) at /data/hf_cache"
echo "state:   $(ls -la $STATE/ 2>/dev/null || echo empty)"
echo "pentad:  $(ls -lh $MIRAGE/Dataset/seeds/pentad_dataset.parquet 2>/dev/null || echo NOT BUILT YET)"
echo "seeds:   $(ls -lh $MIRAGE/Dataset/seeds/*.parquet 2>/dev/null | wc -l) parquet files"

# Restart supervisor (exits after PIPELINE_COMPLETE — must relaunch manually)
pkill -f "supervise_pipeline.sh" 2>/dev/null || true
sleep 1
nohup bash "$REPO/akash/supervise_pipeline.sh" >> /data/logs/supervise.log 2>&1 &
echo "supervisor PID=$!"
echo "CLEAN_AND_LAUNCH_OK"
"""

def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(6):
        try:
            c.connect(HOST, port=PORT, username="root", password=PW, timeout=25, banner_timeout=40)
            break
        except Exception as e:
            print(f"  connect {i+1}: {e}")
            time.sleep(10)
    else:
        print("FAILED to connect"); sys.exit(1)

    print(f"Connected to {HOST}:{PORT}\n")
    _, stdout, stderr = c.exec_command(f"bash -s << 'CLEAN_EOF'\n{CLEAN_SCRIPT}\nCLEAN_EOF", timeout=120)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    print(out)
    if err.strip():
        print("stderr:", err[:500])

    if "CLEAN_AND_LAUNCH_OK" not in out:
        print("WARN: clean script may not have completed cleanly")
        c.close()
        sys.exit(1)

    print("\nWaiting 30s for pipeline to start...")
    time.sleep(30)

    _, stdout, _ = c.exec_command(
        "tail -20 /data/logs/supervise.log 2>/dev/null; "
        "echo '---'; tail -15 $(ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1) 2>/dev/null",
        timeout=30,
    )
    print(stdout.read().decode("utf-8", "replace"))
    c.close()
    print("\nMonitor: python akash/_monitor.py")


if __name__ == "__main__":
    main()
