"""One-shot VM snapshot for slot-b repair."""
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HOST, PORT, PW = "provider.a100.dsm.val.akash.pub", 31532, "MirageVM2026!"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, username="root", password=PW, timeout=30)

SCRIPT = r"""
echo "=== uptime / pid1 ==="
uptime
ps -p 1 -o etimes=

echo "=== processes ==="
pgrep -af regenerate_api_slots || true
pgrep -af supervise_pipeline || true
pgrep -af _full_pipeline || true
pgrep -af run_gpu_pipeline || true

echo "=== markers ==="
ls -la /data/state/

echo "=== regen log tail ==="
tail -5 /data/logs/regenerate_api_slots.log 2>/dev/null || echo no-regen-log

echo "=== pentad ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
p = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet"
df = pd.read_parquet(p)
print("rows", len(df), "slots", df.slot.value_counts().to_dict())
PY
"""

_, stdout, _ = c.exec_command(f"bash -s << 'EOF'\n{SCRIPT}\nEOF", timeout=60)
print(stdout.read().decode("utf-8", "replace"))
c.close()
