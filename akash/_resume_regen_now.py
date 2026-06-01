"""Resume stopped regen from checkpoint; start autonomous guard; no det wipe."""
from __future__ import annotations

import sys
import time

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"

DIAG = r"""
echo "=== container age ==="
ps -p 1 -o etimes=
echo "=== processes ==="
pgrep -af 'supervise|_full_pipeline|run_gpu|regenerate|autonomous_guard' || echo none
echo "=== markers ==="
ls -la /data/state/
echo "=== checkpoint ==="
/data/venv/bin/python - <<'PY'
import json, os
base="/data/Audit_Benchmark/Code/mirage/Dataset/seeds"
for name in ("context_shift_checkpoint.json","cot_attack_checkpoint.json"):
    p=os.path.join(base,name)
    print(name, len(json.load(open(p))) if os.path.exists(p) else 0)
PY
echo "=== det validation ==="
cd /data/Audit_Benchmark/Code/mirage
/data/venv/bin/python - <<'PY'
import pandas as pd
from Dataset.validate_pentad import validate_slot_b_grammar, validate_schema
df = pd.read_parquet("Dataset/seeds/pentad_dataset.parquet")
det = df[df.slot.isin(["a","b","c"])]
validate_schema(det)
validate_slot_b_grammar(det)
print("DET_OK rows", len(det), "seeds", det.seed_id.nunique())
PY
echo "=== regen log tail ==="
tail -8 /data/logs/prelaunch_regen.log 2>/dev/null || true
echo "=== guard log tail ==="
tail -5 /data/logs/autonomous_guard.log 2>/dev/null || echo no guard
"""

RESUME = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1
REPO=/data/Audit_Benchmark
MIRAGE=$REPO/Code/mirage
PY=/data/venv/bin/python
STATE=/data/state
LOG=/data/logs/prelaunch_regen.log

echo "=== stop supervisor/gpu only (keep checkpoints) ==="
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true
sleep 2

echo "=== clear stale downstream markers/results ==="
rm -f $STATE/DATASET_OK $STATE/GPU_PIPELINE_OK $STATE/PIPELINE_COMPLETE
rm -f $MIRAGE/results/behavioral_results.parquet $MIRAGE/results/cdva_results.parquet $MIRAGE/results/tau_calibration.json

if [ -f /data/.env ]; then
  cp /data/.env $MIRAGE/.env
  sed -i 's/\r$//' $MIRAGE/.env
fi

echo "=== resume regen ==="
pkill -f regenerate_api_slots.py 2>/dev/null || true
sleep 2
cd $MIRAGE
if [ -f Dataset/seeds/context_shift_checkpoint.json ] || [ -f Dataset/seeds/cot_attack_checkpoint.json ]; then
  nohup $PY regenerate_api_slots.py --keep-checkpoint >> $LOG 2>&1 &
else
  nohup $PY regenerate_api_slots.py >> $LOG 2>&1 &
fi
echo "regen_pid=$!"

echo "=== start autonomous guard loop ==="
pkill -f 'while true; do bash /data/Audit_Benchmark/akash/autonomous_guard.sh' 2>/dev/null || true
pkill -f 'autonomous_guard.sh' 2>/dev/null || true
sleep 1
sed -i 's/\r$//' $REPO/akash/autonomous_guard.sh
sed -i 's/\r$//' $REPO/akash/supervise_pipeline.sh
chmod +x $REPO/akash/autonomous_guard.sh
nohup bash -c 'while true; do bash /data/Audit_Benchmark/akash/autonomous_guard.sh; sleep 60; done' >> /data/logs/autonomous_guard.log 2>&1 &
echo "guard_pid=$!"

sleep 6
pgrep -af 'regenerate_api|autonomous_guard' || true
tail -3 $LOG || true
"""


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(8):
        try:
            c.connect(HOST, PORT, username=USER, password=PW, timeout=30, banner_timeout=60)
            break
        except Exception as exc:
            print(f"connect {i + 1}: {exc}")
            time.sleep(8)
    else:
        sys.exit(1)

    print("=== DIAGNOSTIC ===")
    code, out, err = run(c, f"bash -s << 'EOF'\n{DIAG}\nEOF", timeout=90)
    print(out)
    if err.strip():
        print("STDERR:", err[-2000:])

    if "DET_OK" not in out:
        print("ABORT: det slots failed validation — manual fix needed")
        c.close()
        sys.exit(1)

    print("\n=== RESUME FROM CHECKPOINT ===")
    code, out, err = run(c, f"bash -s << 'EOF'\n{RESUME}\nEOF", timeout=120)
    print(out)
    if err.strip():
        print("STDERR:", err[-2000:])
    c.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
