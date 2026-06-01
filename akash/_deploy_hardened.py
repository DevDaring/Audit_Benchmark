"""
Deploy all hardened fixes + restart regen with checkpoint (preserves progress).
Starts on-VM autonomous guard loop.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"
REPO, MIRAGE = "/data/Audit_Benchmark", f"/data/Audit_Benchmark/Code/mirage"

UPLOAD = [
    ("Code/mirage/Dataset/pentad_generator.py", f"{MIRAGE}/Dataset/pentad_generator.py"),
    ("Code/mirage/Dataset/validate_pentad.py", f"{MIRAGE}/Dataset/validate_pentad.py"),
    ("Code/mirage/Dataset/context_shift_drafter.py", f"{MIRAGE}/Dataset/context_shift_drafter.py"),
    ("Code/mirage/Dataset/cot_attack_generator.py", f"{MIRAGE}/Dataset/cot_attack_generator.py"),
    ("Code/mirage/regenerate_api_slots.py", f"{MIRAGE}/regenerate_api_slots.py"),
    ("Code/mirage/patch_slot_b_only.py", f"{MIRAGE}/patch_slot_b_only.py"),
    ("Code/mirage/patch_det_slots.py", f"{MIRAGE}/patch_det_slots.py"),
    ("akash/_full_pipeline.py", f"{REPO}/akash/_full_pipeline.py"),
    ("akash/supervise_pipeline.sh", f"{REPO}/akash/supervise_pipeline.sh"),
    ("akash/autonomous_guard.sh", f"{REPO}/akash/autonomous_guard.sh"),
]

REMOTE = r"""
set -euo pipefail
export PYTHONUNBUFFERED=1
PY=/data/venv/bin/python
REPO=/data/Audit_Benchmark
MIRAGE=$REPO/Code/mirage
LOG=/data/logs/prelaunch_regen.log

chmod +x $REPO/akash/autonomous_guard.sh

echo "=== stop supervisor/gpu (keep regen until swap) ==="
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true

if [ -f /data/.env ]; then
  cp /data/.env $MIRAGE/.env
  sed -i 's/\r$//' $MIRAGE/.env
fi

$PY - <<'PY'
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
from config import DEEPSEEK_KEYS
print("deepseek_keys", len(DEEPSEEK_KEYS), all(bool(k) for k in DEEPSEEK_KEYS))
PY

echo "=== checkpoint before swap ==="
$PY - <<'PY'
import json, os
p="/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"
print("ctx_ckpt", len(json.load(open(p))) if os.path.exists(p) else 0)
PY

echo "=== restart regen with hardened code + keep-checkpoint ==="
pkill -f regenerate_api_slots.py 2>/dev/null || true
sleep 3
cd $MIRAGE
nohup $PY regenerate_api_slots.py --keep-checkpoint >> $LOG 2>&1 &
echo "regen_pid=$!"

echo "=== start autonomous guard (every 60s) ==="
pkill -f 'autonomous_guard.sh' 2>/dev/null || true
sleep 1
nohup bash -c 'while true; do bash /data/Audit_Benchmark/akash/autonomous_guard.sh; sleep 60; done' >> /data/logs/autonomous_guard.log 2>&1 &
echo "guard_loop_pid=$!"

sleep 5
pgrep -af 'regenerate_api|autonomous_guard' || true
$PY - <<'PY'
import json, os
p="/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"
print("ctx_ckpt_after", len(json.load(open(p))) if os.path.exists(p) else 0)
PY
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parents[1]

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(10):
        try:
            c.connect(HOST, PORT, username=USER, password=PW, timeout=30, banner_timeout=60)
            break
        except Exception as exc:
            print(f"connect {i+1}: {exc}")
            time.sleep(10)
    else:
        sys.exit(1)

    env = root / "Code" / "mirage" / ".env"
    sftp = c.open_sftp()
    if env.exists():
        sftp.put(str(env), "/data/.env")
        sftp.put(str(env), f"{MIRAGE}/.env")
        print("Uploaded .env")
    for rel, remote in UPLOAD:
        print(f"Upload {rel}")
        sftp.put(str(root / rel), remote)
    sftp.close()

    _, stdout, stderr = c.exec_command(f"bash -s << 'EOF'\n{REMOTE}\nEOF", timeout=120)
    print(stdout.read().decode("utf-8", "replace"))
    err = stderr.read().decode("utf-8", "replace")
    if err.strip():
        print("STDERR:", err[-4000:])
    code = stdout.channel.recv_exit_status()
    c.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
