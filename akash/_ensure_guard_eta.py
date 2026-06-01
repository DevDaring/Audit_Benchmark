"""Ensure guard loop, block supervisor during regen, report ETA."""
from __future__ import annotations

import json
import sys
import time

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"

REMOTE = r"""
set -uo pipefail
echo "=== age ==="
ps -p 1 -o etimes=
echo "=== procs ==="
pgrep -af 'supervise|_full_pipeline|run_gpu|regenerate|autonomous_guard' || echo none
echo "=== markers ==="
ls /data/state/ 2>/dev/null
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true
sed -i 's/\r$//' /data/Audit_Benchmark/akash/autonomous_guard.sh
if pgrep -f regenerate_api_slots.py >/dev/null; then
  if ! pgrep -f 'while true; do bash /data/Audit_Benchmark/akash/autonomous_guard' >/dev/null 2>&1; then
    nohup bash -c 'while true; do bash /data/Audit_Benchmark/akash/autonomous_guard.sh; sleep 60; done' >> /data/logs/autonomous_guard.log 2>&1 &
    echo "started_guard=$!"
  fi
fi
/data/venv/bin/python - <<'PY'
import json, os, pandas as pd
base = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds"
ctx = os.path.join(base, "context_shift_checkpoint.json")
cot = os.path.join(base, "cot_attack_checkpoint.json")
ctx_n = len(json.load(open(ctx))) if os.path.exists(ctx) else 0
cot_n = len(json.load(open(cot))) if os.path.exists(cot) else 0
df = pd.read_parquet(os.path.join(base, "pentad_dataset.parquet"))
de = int(df.slot.isin(["d", "e"]).sum())
print(f"ctx_ckpt {ctx_n}")
print(f"cot_ckpt {cot_n}")
print(f"pentad_rows {len(df)} de {de}")
PY
tail -1 /data/logs/prelaunch_regen.log 2>/dev/null || true
"""


def connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(5):
        try:
            c.connect(HOST, PORT, username=USER, password=PW, timeout=45, banner_timeout=90)
            return c
        except Exception as exc:
            print(f"connect {i + 1}: {exc}")
            time.sleep(15)
    raise SystemExit(1)


def ctx_count(c: paramiko.SSHClient) -> int:
    _, o, _ = c.exec_command(
        "/data/venv/bin/python -c \"import json,os;"
        "p='/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json';"
        "print(len(json.load(open(p))) if os.path.exists(p) else 0)\"",
        timeout=30,
    )
    return int(o.read().decode().strip() or "0")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = connect()
    _, o, _ = c.exec_command(f"bash -s << 'EOF'\n{REMOTE}\nEOF", timeout=90)
    print(o.read().decode())

    n1 = ctx_count(c)
    time.sleep(35)
    try:
        n2 = ctx_count(c)
    except Exception:
        c.close()
        c = connect()
        n2 = ctx_count(c)

    rate_per_min = max(0, (n2 - n1) / (35 / 60))
    ctx_left = max(0, 596 - n2)
    ctx_min = ctx_left / rate_per_min if rate_per_min > 0 else 15
    # slot-e ~596 seeds similar rate + validation + GPU ~6h
    regen_min = ctx_min + 20  # slot-e buffer
    gpu_h = 6.0
    total_h = regen_min / 60 + gpu_h

    print("\n=== ETA ===")
    print(f"ctx_ckpt: {n1} -> {n2} in 35s (~{rate_per_min:.0f} seeds/min)")
    print(f"slot-d remaining: ~{ctx_left} seeds (~{ctx_min:.0f} min)")
    print(f"regen phase ETA: ~{regen_min:.0f} min (slot-d + slot-e)")
    print(f"GPU phase ETA: ~{gpu_h:.0f} h after dataset ready")
    print(f"TOTAL ETA: ~{total_h:.1f} h")

    c.close()


if __name__ == "__main__":
    main()
