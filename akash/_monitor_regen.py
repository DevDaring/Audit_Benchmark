"""Monitor VM DeepSeek regeneration progress."""
import json
import pathlib
import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("provider.a100.dsm.val.akash.pub", 31532, username="root", password="MirageVM2026!", timeout=25)

log = "/data/Audit_Benchmark/LOG/regen_api_slots.log"
cp = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"

for cmd in [
    f"grep -c 'Context-shift OK' {log} 2>/dev/null || echo 0",
    f"grep -c FAILED {log} 2>/dev/null || echo 0",
    f"tail -12 {log}",
    f"test -f {cp} && wc -c {cp} || echo no-checkpoint",
    "pgrep -af regenerate_api_slots || echo not-running",
]:
    _, o, _ = c.exec_command(cmd, timeout=20)
    print("===", cmd[:60], "===")
    print(o.read().decode("utf-8", "replace"))

c.close()
