"""Deep VM check: progress delta, disk, errors since GPU start."""
import re
import time
from datetime import datetime

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode("utf-8", "replace").strip()


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)

    print("=== DISK ===")
    print(run(c, "df -h / /data /workspace 2>/dev/null; df -i / /data 2>/dev/null | tail -2"))

    print("\n=== PROGRESS SAMPLE 1 ===")
    p1 = run(c, 'grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -1')
    print(p1)
    time.sleep(30)
    print("\n=== PROGRESS SAMPLE 2 (30s later) ===")
    p2 = run(c, 'grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -1')
    print(p2)

    m1 = re.search(r": (\d+)/7152", p1)
    m2 = re.search(r": (\d+)/7152", p2)
    if m1 and m2:
        delta = int(m2.group(1)) - int(m1.group(1))
        print(f"\nProgress in 30s: +{delta} prompts ({delta * 2}/min approx)")

    ln = run(c, 'grep -n "Step 2/4: Behavioral" /data/logs/pipeline_attempt_1.log | tail -1').split(":")[0]
    if ln:
        print("\n=== ERRORS SINCE GPU STEP2 ===")
        print(
            run(
                c,
                f"tail -n +{ln} /data/logs/pipeline_attempt_1.log | "
                "grep -iE 'traceback|error:|failed|exception' | grep -v 'failure_reason' | head -10 || echo NONE",
            )
        )

    print("\n=== WATCHDOG (last 3 lines) ===")
    print(run(c, "tail -3 /data/logs/watchdog.log 2>/dev/null || echo no watchdog"))

    print("\n=== REGEN RUNNING? ===")
    print(run(c, "pgrep -af regenerate_api || echo not running"))

    c.close()


if __name__ == "__main__":
    main()
