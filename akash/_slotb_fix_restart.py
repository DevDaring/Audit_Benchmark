"""
Stop GPU, clear dataset/GPU markers and stale results, upload slot-b fix, patch pentad, restart.
"""
import sys
import time

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"
REPO = "/data/Audit_Benchmark"
MIRAGE = f"{REPO}/Code/mirage"

UPLOAD = [
    ("Code/mirage/Dataset/pentad_generator.py", f"{MIRAGE}/Dataset/pentad_generator.py"),
    ("Code/mirage/Dataset/validate_pentad.py", f"{MIRAGE}/Dataset/validate_pentad.py"),
    ("Code/mirage/patch_slot_b_only.py", f"{MIRAGE}/patch_slot_b_only.py"),
    ("Code/mirage/regenerate_api_slots.py", f"{MIRAGE}/regenerate_api_slots.py"),
]

REMOTE_SCRIPT = r"""
set -euo pipefail
echo "=== 1. STOP everything (supervisor first — prevents patch_det_slots wipe) ==="
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f regenerate_api_slots.py 2>/dev/null || true
sleep 5
pgrep -af 'supervise|run_gpu|_full_pipeline|regenerate' || echo "all pipeline processes stopped"

echo "=== 2. CLEAR markers (dataset + GPU; keep install/predownload) ==="
rm -f /data/state/DATASET_OK /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE

echo "=== 3. CLEAR stale GPU results ==="
rm -f /data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/tau_calibration.json
rm -f /data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json
rm -f /data/Audit_Benchmark/Code/mirage/Dataset/seeds/cot_attack_checkpoint.json
ls -la /data/state/

echo "=== 4. PATCH slot-b (keeps existing d/e if present) ==="
cd /data/Audit_Benchmark/Code/mirage
/data/venv/bin/python patch_slot_b_only.py
echo "patch slot-b exit: $?"

echo "=== 5. REGEN d/e if missing (DeepSeek) ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
df = pd.read_parquet(p)
n_de = (df['slot'].isin(['d','e'])).sum()
print('d/e rows before regen:', n_de)
PY
/data/venv/bin/python regenerate_api_slots.py
echo "regen exit: $?"

echo "=== 6. FINAL VALIDATION ==="
/data/venv/bin/python - <<'PY'
import pandas as pd, sys
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar, write_pentad_manifest
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
print('rows', len(df), 'slots', df.slot.value_counts().to_dict())
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

echo "=== 7. MARK DATASET_OK ==="
touch /data/state/DATASET_OK

echo "=== 8. RESTART supervisor ==="
nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
echo "supervisor pid=$!"
sleep 8
pgrep -af supervise_pipeline || true
tail -8 /data/logs/supervise.log 2>/dev/null || true
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = __import__("pathlib").Path(__file__).resolve().parents[1]

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)

    sftp = c.open_sftp()
    for local_rel, remote in UPLOAD:
        local = root / local_rel
        print(f"Upload {local_rel} -> {remote}")
        sftp.put(str(local), remote)
    sftp.close()

    print("\nRunning stop / patch / restart on VM...\n")
    _, stdout, stderr = c.exec_command(f"bash -s << 'EOF'\n{REMOTE_SCRIPT}\nEOF", timeout=7200)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    print(out)
    if err.strip():
        print("STDERR:", err[-3000:])
    code = stdout.channel.recv_exit_status()
    c.close()
    if code != 0:
        sys.exit(code)
    print("\nDone. Pipeline restarted with fixed slot-b pentad.")


if __name__ == "__main__":
    main()
