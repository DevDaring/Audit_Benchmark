"""Poll regen then validate, clean, launch (run after phase 1 started regen)."""
import sys
from pathlib import Path

# reuse main script phases
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _prelaunch_audit_clean import POLL, PHASE3, connect, run, main as _unused

import argparse
import time


def finish_only(no_launch: bool = False) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    launch_flag = "0" if no_launch else "1"

    print("=== Phase 2: wait for regen (up to 2h) ===")
    deadline = time.time() + 43200
    while time.time() < deadline:
        try:
            c = connect()
            code, out, err = run(c, POLL, timeout=60)
            c.close()
        except Exception as exc:
            print(f"  poll error: {exc}")
            time.sleep(20)
            continue

        print(time.strftime("%H:%M:%S"), out.strip().replace("\n", " | "))
        if "DONE" in out:
            for line in out.splitlines():
                if line.startswith("rows") and "de" in line:
                    parts = line.split()
                    try:
                        de = int(parts[parts.index("de") + 1])
                        need = int(parts[parts.index("need") + 1])
                        if de >= need and de > 0:
                            print("Regen complete.")
                            break
                    except (ValueError, IndexError):
                        pass
            else:
                time.sleep(45)
                continue
            break
        time.sleep(45)
    else:
        print("TIMEOUT")
        sys.exit(1)

    print("\n=== Phase 3: validate + clean + launch ===")
    c = connect()
    phase3 = PHASE3.replace("__LAUNCH__", launch_flag)
    code, out, err = run(c, phase3, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err[-6000:])
    c.close()
    if code != 0 or "PRELAUNCH_OK" not in out:
        sys.exit(code or 1)
    print("\nVM is production-ready.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-launch", action="store_true")
    a = p.parse_args()
    finish_only(a.no_launch)
