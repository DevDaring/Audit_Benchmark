"""Verify VM env keys and regen health without printing secrets."""
import paramiko
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!", timeout=30)

SCRIPT = r"""
/data/venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
# load .env from mirage
from pathlib import Path
env = Path("/data/Audit_Benchmark/Code/mirage/.env")
if not env.exists():
    env = Path("/data/.env")
for line in env.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
from config import DEEPSEEK_KEYS, DEEPSEEK_API_BASE_URL, DEEPSEEK_PRIMARY_MODEL_NAME
print("keys_loaded", len(DEEPSEEK_KEYS), "all_nonempty", all(bool(k) for k in DEEPSEEK_KEYS))
print("keys_distinct", len(set(DEEPSEEK_KEYS)) == len(DEEPSEEK_KEYS))
print("base_url", DEEPSEEK_API_BASE_URL)
print("model", DEEPSEEK_PRIMARY_MODEL_NAME)
PY
grep -c "Context-shift OK" /data/logs/prelaunch_regen.log 2>/dev/null || echo 0
ls -la /data/Audit_Benchmark/Code/mirage/.env /data/.env 2>/dev/null
"""

_, o, _ = c.exec_command(f"bash -s << 'EOF'\n{SCRIPT}\nEOF", timeout=30)
print(o.read().decode())
c.close()
