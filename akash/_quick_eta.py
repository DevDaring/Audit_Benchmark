"""Quick ETA from latest GPU progress."""
import re
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

    print("=== LATEST PROGRESS ===")
    print(run(c, 'grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -3'))

    start = run(c, 'grep -n "Step 2/4: Behavioral" /data/logs/pipeline_attempt_1.log | tail -1')
    print("\n=== GPU STEP2 START ===")
    print(start)

    if start:
        ln = start.split(":")[0]
        errs = run(
            c,
            f"tail -n +{ln} /data/logs/pipeline_attempt_1.log | "
            "grep -E 'Traceback|FAILED|ERROR' | head -5 || echo NO_ERRORS_SINCE_GPU_START",
        )
        print("\n=== ERRORS SINCE GPU STEP2 ===")
        print(errs)

    lines = run(c, 'grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -10').split("\n")
    pts = []
    for line in lines:
        m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z.*?: (\d+)/7152", line)
        if m:
            pts.append((datetime.fromisoformat(m.group(1)), int(m.group(2))))

    if len(pts) >= 2:
        dt_min = (pts[-1][0] - pts[0][0]).total_seconds() / 60
        dn = pts[-1][1] - pts[0][1]
        rate = dn / dt_min if dt_min > 0 else 0
        done = pts[-1][1]
        # model 1 det remaining + 3 models full + all variance + CDVA
        remaining_det_m1 = 7152 - done
        remaining_det_other = 3 * 7152
        remaining_var = 4 * 596 * 5
        remaining = remaining_det_m1 + remaining_det_other + remaining_var
        gpu_hours = remaining / rate / 60 if rate > 0 else 0
        total_hours = gpu_hours + 1.5
        print(f"\n=== RATE & ETA ===")
        print(f"Current: {done}/7152 (model 1 deterministic)")
        print(f"Rate: {rate:.1f} prompts/min (last 10 log lines)")
        print(f"Remaining prompts: {remaining}")
        print(f"GPU work ETA: {gpu_hours:.1f} h")
        print(f"Total ETA (incl CDVA/calibration): {total_hours:.1f} h")

    c.close()


if __name__ == "__main__":
    main()
