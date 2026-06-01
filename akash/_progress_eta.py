"""Progress snapshot + ETA for regen and full pipeline."""
import json
import re
import sys
import time
from datetime import datetime, timezone

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"


def run(c, script, timeout=60):
    _, o, _ = c.exec_command(f"bash -s << 'EOF'\n{script}\nEOF", timeout=timeout)
    return o.read().decode("utf-8", "replace")


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PW, timeout=30, banner_timeout=60)
    return c


def snapshot(c):
    return run(
        c,
        r"""
PY=/data/venv/bin/python
echo "=== MARKERS ==="
ls -la /data/state/ 2>/dev/null
echo "=== PROCS ==="
pgrep -af 'supervise_pipeline|_full_pipeline|run_gpu|regenerate_api' || echo none
echo "=== REGEN ==="
pgrep -f regenerate_api_slots.py >/dev/null && echo RUNNING || echo DONE
$PY - <<'PY'
import json, os, pandas as pd
from pathlib import Path
p = Path("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
df = pd.read_parquet(p)
de = int(df.slot.isin(["d","e"]).sum())
ck = Path("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json")
cot = Path("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/cot_attack_checkpoint.json")
print("pentad_rows", len(df), "de_rows", de)
print("ctx_ckpt", len(json.load(open(ck))) if ck.exists() else 0)
print("cot_ckpt", len(json.load(open(cot))) if cot.exists() else 0)
PY
echo "=== LOG TAIL ==="
tail -2 /data/logs/prelaunch_regen.log 2>/dev/null || true
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
""",
    )


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = connect()
    s1 = snapshot(c)
    print("T1", datetime.now(timezone.utc).strftime("%H:%M:%S"))
    print(s1)
    m1 = re.search(r"ctx_ckpt (\d+)", s1)
    ck1 = int(m1.group(1)) if m1 else 0
    time.sleep(60)
    s2 = snapshot(c)
    print("\nT2 (+60s)", datetime.now(timezone.utc).strftime("%H:%M:%S"))
    print(s2)
    c.close()

    m2 = re.search(r"ctx_ckpt (\d+)", s2)
    ck2 = int(m2.group(1)) if m2 else 0
    rate = max(0, ck2 - ck1)  # seeds per minute

    regen_running = "RUNNING" in s2 and "regenerate_api_slots" in s2
    de_match = re.search(r"de_rows (\d+)", s2)
    de = int(de_match.group(1)) if de_match else 0

    print("\n=== ANALYSIS ===")
    if de >= 2980:
        print("Regen: COMPLETE (d/e in pentad)")
        regen_min = 0
    elif regen_running:
        ctx_left = max(0, 596 - ck2)
        cot_left = 596
        if rate > 0:
            ctx_min = ctx_left / rate
            cot_min = cot_left / rate  # similar rate assumed
            regen_min = ctx_min + cot_min
            print(f"Regen: IN PROGRESS ctx={ck2}/596 rate~{rate}/min")
            print(f"  ETA regen: ~{regen_min:.0f} min ({regen_min/60:.1f} h)")
        else:
            regen_min = 90
            print(f"Regen: IN PROGRESS ctx={ck2}/596 (rate unknown, est ~90 min)")
    else:
        print("Regen: NOT RUNNING — needs restart")
        regen_min = None

    gpu_hours = 5.5  # 4 models behavioral + CDVA from prior calibration
    if regen_min is not None:
        total_h = regen_min / 60 + gpu_hours
        print(f"\nFull pipeline ETA: ~{total_h:.1f} hours from now")
        print(f"  ({regen_min/60:.1f}h regen + ~{gpu_hours}h GPU/CDVA)")


if __name__ == "__main__":
    main()
