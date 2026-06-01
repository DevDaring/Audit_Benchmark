"""Wait for regen, validate fixed pentad, set DATASET_OK, restart supervisor."""
import sys
import time

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"

STOP = r"""
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true
sleep 3
pgrep -af 'supervise|_full_pipeline|run_gpu' || echo stopped
"""

WAIT_REGEN = r"""
for i in $(seq 1 180); do
  if pgrep -f 'regenerate_api_slots.py' >/dev/null 2>&1; then
    n=$(grep -c 'Context-shift OK' /data/logs/regenerate_api_slots.log 2>/dev/null || echo 0)
    echo "regen running... context_shift_ok=$n"
    sleep 30
  else
    echo "regen not running"
    break
  fi
done
tail -5 /data/logs/regenerate_api_slots.log 2>/dev/null || tail -5 /data/Audit_Benchmark/LOG/regen_api_slots.log 2>/dev/null || true
"""

FINISH = r"""
set -euo pipefail
cd /data/Audit_Benchmark/Code/mirage

echo "=== pentad shape ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
p = '/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet'
df = pd.read_parquet(p)
print('rows', len(df), 'slots', df.slot.value_counts().to_dict())
PY

echo "=== FINAL VALIDATION ==="
/data/venv/bin/python - <<'PY'
import pandas as pd, sys
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar, write_pentad_manifest
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
validate_slot_b_grammar(df)
assert_production_ready(df)
write_pentad_manifest(df)
flags = 0
audit = df[df.seed_source.str.lower().isin(['bbq','crows_pairs','stereoset'])]
for sid, g in audit.groupby('seed_id'):
    t = str(g[g.slot=='b'].iloc[0]['prompt_text']).lower()
    if 'person and person' in t or 'person man' in t or 'a person man' in t:
        flags += 1
print('grammar_flags', flags)
if flags:
    sys.exit(1)
print('VALIDATION_OK')
PY

rm -f /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
rm -f /data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/tau_calibration.json
touch /data/state/DATASET_OK
ls -la /data/state/

echo "=== RESTART supervisor ==="
nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
sleep 6
pgrep -af supervise_pipeline || true
tail -6 /data/logs/supervise.log
"""


def run(client, script, timeout=7200):
    _, stdout, stderr = client.exec_command(f"bash -s << 'EOF'\n{script}\nEOF", timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)

    print("Stopping supervisor / GPU (regen continues)...")
    code, out, err = run(c, STOP, timeout=60)
    print(out)
    if err.strip():
        print("STDERR:", err[-1500:])

    print("\nWaiting for regenerate_api_slots.py to finish (up to 90 min)...")
    code, out, err = run(c, WAIT_REGEN, timeout=7200)
    print(out)

    print("\nValidate + DATASET_OK + restart...")
    code, out, err = run(c, FINISH, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err[-3000:])
    c.close()
    if code != 0:
        sys.exit(code)
    print("\nDone — pipeline restarted with validated fixed pentad.")


if __name__ == "__main__":
    main()
