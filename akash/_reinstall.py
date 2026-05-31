"""
akash/_reinstall.py
Bootstrap pip correctly on the Akash VM and install all MIRAGE packages.

Root cause of the previous install failure:
  Ubuntu 22.04 ships a patched python3-pip.  When that pip is upgraded in-place
  via "pip install --upgrade pip", the apt-managed pip module disappears from
  the system site-packages and "python3 -m pip" breaks (No module named pip).
  The fixed install.sh avoids this by bootstrapping from get-pip.py first,
  but we must also fix the running VM.

This script:
  1. SSH into the VM.
  2. Run the fixed install.sh inside a tmux session.
  3. Poll until DONE or FAILED.
  4. Print the last 30 log lines.

Usage (from repo root on Windows):
  python akash/_reinstall.py
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

# ---------------------------------------------------------------------------
# VM SSH coordinates — edit to match akash/vm_ssh.txt
# ---------------------------------------------------------------------------
VM_HOST = os.getenv("AKASH_VM_HOST", "provider.a100.dsm.val.akash.pub")
VM_PORT = int(os.getenv("AKASH_VM_PORT", "30594"))
VM_USER = os.getenv("AKASH_VM_USER", "root")
VM_KEY_PATH = os.getenv("AKASH_VM_KEY", "")  # leave empty for password auth
VM_PASSWORD = os.getenv("AKASH_VM_PASSWORD", "")  # used if no key

POLL_INTERVAL = 30   # seconds between log-tail polls
MAX_WAIT_MIN  = 60   # abort if not done within 60 minutes


def _connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kw: dict = dict(
        hostname=VM_HOST,
        port=VM_PORT,
        username=VM_USER,
        timeout=30,
        banner_timeout=60,
    )
    if VM_KEY_PATH and Path(VM_KEY_PATH).exists():
        connect_kw["key_filename"] = VM_KEY_PATH
    elif VM_PASSWORD:
        connect_kw["password"] = VM_PASSWORD
    else:
        # Try default keys in ~/.ssh/
        pass
    client.connect(**connect_kw)
    return client


def _run(client: paramiko.SSHClient, cmd: str, timeout: int = 30) -> str:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return (out + "\n" + err).strip()


def main() -> None:
    print(f"Connecting to {VM_USER}@{VM_HOST}:{VM_PORT} ...")
    client = _connect()
    print("Connected.")

    # Ensure the updated install.sh is present (pulled from git).
    print("\n[1] git pull to get the fixed install.sh ...")
    pull_out = _run(client, "cd /workspace/Audit_Benchmark && git pull 2>&1", timeout=60)
    print(pull_out[:500])

    # Kill any stale tmux install session.
    _run(client, "tmux kill-session -t reinstall 2>/dev/null; true")

    # Start fresh install in tmux so SSH timeout won't kill it.
    LOG = "/workspace/reinstall.log"
    tmux_cmd = (
        f'tmux new-session -d -s reinstall '
        f'"bash /workspace/Audit_Benchmark/akash/install.sh 2>&1 | tee {LOG}; '
        f'echo REINSTALL_DONE >> {LOG}"'
    )
    print(f"\n[2] Launching install.sh inside tmux (log: {LOG}) ...")
    _run(client, tmux_cmd)
    print("  tmux session 'reinstall' started.")

    # Poll.
    deadline = time.time() + MAX_WAIT_MIN * 60
    polls = 0
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        polls += 1
        done_count = _run(client, f"grep -c REINSTALL_DONE {LOG} 2>/dev/null || echo 0")
        print(f"  [{polls * POLL_INTERVAL}s] REINSTALL_DONE count: {done_count.split()[-1]}")

        if done_count.strip().endswith("1"):
            break
    else:
        print("TIMEOUT: install took too long.  Check /workspace/reinstall.log on the VM.")
        client.close()
        sys.exit(1)

    # Print verification block.
    print("\n[3] Last 40 lines of reinstall.log:")
    tail = _run(client, f"tail -40 {LOG}", timeout=30)
    print(tail)

    # Quick sanity: can we import torch?
    print("\n[4] torch import check:")
    torch_check = _run(
        client,
        "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1",
        timeout=30,
    )
    print(torch_check)

    print("\n[5] dotenv import check:")
    dot_check = _run(
        client,
        "python3 -c 'import dotenv; print(dotenv.__version__)' 2>&1",
        timeout=15,
    )
    print(dot_check)

    client.close()
    all_ok = "True" in torch_check and "error" not in dot_check.lower()
    if all_ok:
        print("\nReinstall SUCCEEDED. VM is ready for the dry run.")
    else:
        print("\nReinstall FAILED or packages still broken. Review the log above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
