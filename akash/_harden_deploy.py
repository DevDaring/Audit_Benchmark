"""
Upload hardened code + .env, restart regen with parallel DeepSeek workers (keep checkpoint).
"""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"
REPO = "/data/Audit_Benchmark"
MIRAGE = f"{REPO}/Code/mirage"

UPLOAD = [
    ("Code/mirage/Dataset/pentad_generator.py", f"{MIRAGE}/Dataset/pentad_generator.py"),
    ("Code/mirage/Dataset/validate_pentad.py", f"{MIRAGE}/Dataset/validate_pentad.py"),
    ("Code/mirage/Dataset/context_shift_drafter.py", f"{MIRAGE}/Dataset/context_shift_drafter.py"),
    ("Code/mirage/Dataset/cot_attack_generator.py", f"{MIRAGE}/Dataset/cot_attack_generator.py"),
    ("Code/mirage/patch_slot_b_only.py", f"{MIRAGE}/patch_slot_b_only.py"),
    ("Code/mirage/regenerate_api_slots.py", f"{MIRAGE}/regenerate_api_slots.py"),
    ("Code/mirage/patch_det_slots.py", f"{MIRAGE}/patch_det_slots.py"),
    ("akash/_full_pipeline.py", f"{REPO}/akash/_full_pipeline.py"),
]

REMOTE = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1
PY=/data/venv/bin/python
LOG=/data/logs/prelaunch_regen.log

echo "=== stop supervisor/gpu only ==="
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true

echo "=== sync .env ==="
if [ -f /data/.env ]; then
  cp /data/.env /data/Audit_Benchmark/Code/mirage/.env
  sed -i 's/\r$//' /data/Audit_Benchmark/Code/mirage/.env
fi

echo "=== verify keys (no secrets printed) ==="
$PY - <<'PY'
import os, sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
from config import DEEPSEEK_KEYS
print("keys", len(DEEPSEEK_KEYS), "ok", all(bool(k) for k in DEEPSEEK_KEYS))
PY

echo "=== restart regen with keep-checkpoint + parallel workers ==="
pkill -f regenerate_api_slots.py 2>/dev/null || true
sleep 2
cd /data/Audit_Benchmark/Code/mirage
nohup $PY regenerate_api_slots.py --keep-checkpoint >> $LOG 2>&1 &
echo "regen_pid=$!"
sleep 3
pgrep -af regenerate_api_slots || true
/data/venv/bin/python - <<'PY'
import json, os
p="/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"
print("ctx_ckpt", len(json.load(open(p))) if os.path.exists(p) else 0)
PY
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parents[1]

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(8):
        try:
            c.connect(HOST, PORT, username=USER, password=PW, timeout=30, banner_timeout=60)
            break
        except Exception as exc:
            print(f"connect {i+1}: {exc}")
            import time
            time.sleep(10)
    else:
        sys.exit("Cannot connect")

    # upload .env to persistent volume
    env_local = root / "Code" / "mirage" / ".env"
    if env_local.exists():
        sftp = c.open_sftp()
        sftp.put(str(env_local), "/data/.env")
        sftp.put(str(env_local), f"{MIRAGE}/.env")
        sftp.close()
        print("Uploaded .env")

    sftp = c.open_sftp()
    for local_rel, remote in UPLOAD:
        print(f"Upload {local_rel}")
        sftp.put(str(root / local_rel), remote)
    sftp.close()

    _, stdout, stderr = c.exec_command(f"bash -s << 'EOF'\n{REMOTE}\nEOF", timeout=120)
    print(stdout.read().decode("utf-8", "replace"))
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        print("STDERR:", err[-3000:])
    code = stdout.channel.recv_exit_status()
    c.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
