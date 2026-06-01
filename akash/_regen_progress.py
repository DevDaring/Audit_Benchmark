"""Check regen progress."""
import paramiko
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!", timeout=30)

SCRIPT = r"""
echo "=== regen proc ==="
pgrep -af regenerate_api_slots || echo none
echo "=== prelaunch log ==="
wc -l /data/logs/prelaunch_regen.log 2>/dev/null || echo 0
tail -4 /data/logs/prelaunch_regen.log 2>/dev/null || true
echo "=== checkpoint ==="
/data/venv/bin/python - <<'PY'
import json, os
p = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"
print("ctx_ckpt", len(json.load(open(p))) if os.path.exists(p) else 0)
PY
"""

_, o, _ = c.exec_command(f"bash -s << 'EOF'\n{SCRIPT}\nEOF", timeout=30)
print(o.read().decode())
c.close()
