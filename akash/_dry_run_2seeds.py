"""
akash/_dry_run_2seeds.py
Pull latest code to the Akash VM and launch a GPU dry run with exactly
2 seeds inside a tmux session, then tail the output.

Usage:
  python akash/_dry_run_2seeds.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import paramiko  # type: ignore
except ImportError:
    print("paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

VM_HOST     = os.getenv("AKASH_VM_HOST", "provider.a100.dsm.val.akash.pub")
VM_PORT     = int(os.getenv("AKASH_VM_PORT", "30594"))
VM_USER     = os.getenv("AKASH_VM_USER", "root")
VM_KEY_PATH = os.getenv("AKASH_VM_KEY", "")
VM_PASSWORD = os.getenv("AKASH_VM_PASSWORD", "")

DRY_RUN_LOG   = "/workspace/dry_run_2seeds.log"
TMUX_SESSION  = "dry2"
POLL_INTERVAL = 15   # seconds
MAX_WAIT_MIN  = 30   # abort if still running after 30 min


def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw: dict = dict(hostname=VM_HOST, port=VM_PORT, username=VM_USER,
                    timeout=30, banner_timeout=60)
    if VM_KEY_PATH and Path(VM_KEY_PATH).exists():
        kw["key_filename"] = VM_KEY_PATH
    elif VM_PASSWORD:
        kw["password"] = VM_PASSWORD
    client.connect(**kw)
    return client


def _run(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return (out + "\n" + err).strip()


def main() -> None:
    print(f"Connecting to {VM_USER}@{VM_HOST}:{VM_PORT} ...")
    client = _connect()
    print("Connected.\n")

    # Pull latest code.
    print("[1] git pull ...")
    print(_run(client, "cd /workspace/Audit_Benchmark && git pull 2>&1", timeout=60))

    # Kill any stale dry-run tmux session.
    _run(client, f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null; true")

    # Build the command that runs inside tmux.
    # PYTHONPATH includes mirage/ so imports resolve correctly.
    run_cmd = (
        f"cd /workspace/Audit_Benchmark/Code/mirage && "
        f"PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage "
        f"python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2 2>&1 | tee {DRY_RUN_LOG}; "
        f"echo DRY_RUN_DONE >> {DRY_RUN_LOG}"
    )
    tmux_launch = (
        f'tmux new-session -d -s {TMUX_SESSION} "{run_cmd}"'
    )
    print(f"\n[2] Launching dry run (n_seeds=2) inside tmux session '{TMUX_SESSION}' ...")
    _run(client, tmux_launch)
    print(f"  Log: {DRY_RUN_LOG}")

    client.close()

    # Poll by reconnecting each time to avoid SSH session timeout.
    deadline = time.time() + MAX_WAIT_MIN * 60
    polls = 0
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        polls += 1
        try:
            cl2 = _connect()
            done = _run(cl2, f"grep -c DRY_RUN_DONE {DRY_RUN_LOG} 2>/dev/null || echo 0")
            tail = _run(cl2, f"tail -5 {DRY_RUN_LOG}", timeout=15)
            cl2.close()
        except Exception as exc:
            print(f"  [{polls * POLL_INTERVAL}s] reconnect error: {exc}")
            continue
        print(f"\n  [{polls * POLL_INTERVAL}s] DONE={done.split()[-1]}")
        print("  " + "\n  ".join(tail.splitlines()))
        if done.strip().endswith("1"):
            break
    else:
        print("\nTIMEOUT. Review the log on the VM.")
        sys.exit(1)

    # Final read via fresh connection.
    client = _connect()

    print("\n" + "=" * 60)
    print("FULL DRY RUN LOG:")
    print("=" * 60)
    print(_run(client, f"cat {DRY_RUN_LOG}", timeout=60))

    fails = _run(client, f"grep -c FAIL {DRY_RUN_LOG} 2>/dev/null || echo 0")
    passes = _run(client, f"grep -c PASS {DRY_RUN_LOG} 2>/dev/null || echo 0")
    print(f"\nRESULT: {passes.split()[-1]} PASS, {fails.split()[-1]} FAIL lines in log.")

    client.close()
    if int(fails.split()[-1]) == 0:
        print("Dry run PASSED.")
    else:
        print("Dry run had FAILs. Review the log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
