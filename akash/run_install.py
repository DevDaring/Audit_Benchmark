"""
akash/run_install.py — Remote orchestrator for the MIRAGE Akash VM.

Usage:
    python akash/run_install.py --host <IP> --port <PORT> [--action ACTION]

Actions:
    install      Run install.sh on the VM (idempotent; safe to re-run)
    upload_env   SCP the local .env to the VM
    dry_run      Run the full dry_run_all.py suite and report results
    gpu_run      Launch the full GPU pipeline in a tmux session
    status       Show GPU memory, tmux sessions, and tail of install log
    all          upload_env → install → dry_run (default)

The script streams SSH output in real time so you can watch progress.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import paramiko  # type: ignore
except ImportError:
    print("paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "Code" / "mirage" / ".env"
REMOTE_REPO = "/workspace/Audit_Benchmark"
REMOTE_CODE = f"{REMOTE_REPO}/Code/mirage"


# ------------------------------------------------------------------ SSH helpers

def _connect(host: str, port: int, user: str = "root",
             password: str = "MirageVM2026!", retries: int = 10) -> paramiko.SSHClient:
    """Connect with retries (useful right after container boots)."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, retries + 1):
        try:
            client.connect(host, port=port, username=user, password=password,
                           timeout=30, banner_timeout=60)
            print(f"[run] SSH connected to {host}:{port}")
            return client
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"[run] SSH attempt {attempt}/{retries} failed ({exc}); retrying in 15 s ...")
            time.sleep(15)
    raise RuntimeError("SSH connection failed after all retries")  # never reached


def _stream(client: paramiko.SSHClient, cmd: str,
            timeout: Optional[int] = None) -> int:
    """Execute cmd and stream stdout/stderr to terminal. Returns exit code."""
    transport = client.get_transport()
    chan = transport.open_session()
    chan.set_combine_stderr(True)
    if timeout:
        chan.settimeout(timeout)
    chan.exec_command(cmd)
    while True:
        chunk = chan.recv(4096)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
    return chan.recv_exit_status()


def _run(client: paramiko.SSHClient, cmd: str) -> tuple[int, str]:
    """Execute cmd silently. Returns (exit_code, stdout+stderr)."""
    _, stdout, stderr = client.exec_command(cmd, timeout=120)
    ec = stdout.channel.recv_exit_status()
    out = stdout.read().decode() + stderr.read().decode()
    return ec, out


# ------------------------------------------------------------------ actions

def action_upload_env(host: str, port: int) -> None:
    """SCP the .env file to the VM."""
    if not ENV_FILE.exists():
        print(f"[run] ERROR: {ENV_FILE} not found.")
        sys.exit(1)

    client = _connect(host, port)
    sftp = client.open_sftp()

    # Ensure remote directory exists
    _run(client, f"mkdir -p {REMOTE_CODE}")

    remote_env = f"{REMOTE_CODE}/.env"
    sftp.put(str(ENV_FILE), remote_env)
    _run(client, f"chmod 600 {remote_env}")
    print(f"[run] .env uploaded to {remote_env}")

    sftp.close()
    client.close()


def action_install(host: str, port: int) -> None:
    """Run install.sh on the VM, streaming output."""
    client = _connect(host, port)
    print("[run] === Running install.sh (this will take ~5-10 minutes) ===")
    rc = _stream(client, f"bash {REMOTE_REPO}/akash/install.sh")
    client.close()
    if rc != 0:
        print(f"[run] install.sh FAILED (exit {rc})")
        sys.exit(rc)
    print("[run] install.sh PASSED")


def action_dry_run(host: str, port: int) -> None:
    """Run dry_run_all.py and report pass/fail."""
    client = _connect(host, port)
    print("[run] === Running dry_run_all.py ===")
    rc = _stream(
        client,
        f"cd {REMOTE_CODE} && python3 Dry_Run/dry_run_all.py",
        timeout=600,
    )
    client.close()
    if rc != 0:
        print(f"[run] dry_run_all.py FAILED (exit {rc})")
        sys.exit(rc)
    print("[run] dry_run_all.py PASSED")


def action_gpu_run(host: str, port: int) -> None:
    """
    Launch the full GPU pipeline inside a tmux session on the VM.

    Runs sequentially:
      1. osm_behavioral.py  (all 4 models × all prompts)
      2. cdva_patching.py   (TL + nnsight activation patching)
      3. cdva_calibration.py (tau calibration)
      4. leaderboard.py      (aggregate results)

    Output goes to /workspace/logs/gpu_run.log and is also shown in tmux.
    """
    client = _connect(host, port)
    tmux_cmd = (
        "tmux new-session -d -s gpu_run || true; "
        "tmux send-keys -t gpu_run "
        f"'cd {REMOTE_CODE} && "
        "export HF_HOME=/workspace/.hf_cache && "
        "bash -c \""
        "python3 GPU_CPU/osm_behavioral.py && "
        "python3 GPU_CPU/cdva_patching.py && "
        "python3 GPU_CPU/cdva_calibration.py && "
        "python3 CPU_Only/leaderboard.py"
        "\" "
        f"2>&1 | tee /workspace/logs/gpu_run.log' Enter"
    )
    rc, out = _run(client, tmux_cmd)
    client.close()
    if rc != 0:
        print(f"[run] tmux launch FAILED: {out}")
        sys.exit(rc)
    print("[run] GPU pipeline launched in tmux session 'gpu_run'.")
    print(f"[run] Attach:  ssh root@{host} -p {port}  then  tmux attach -t gpu_run")
    print(f"[run] Log:     /workspace/logs/gpu_run.log")


def action_status(host: str, port: int) -> None:
    """Show GPU memory, tmux sessions, and last 30 lines of install log."""
    client = _connect(host, port)
    sections = [
        ("nvidia-smi (memory)", "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv"),
        ("tmux sessions", "tmux ls 2>/dev/null || echo 'no sessions'"),
        ("install log (last 20 lines)", "tail -20 /workspace/install.log 2>/dev/null || echo 'no log'"),
        ("VRAM usage detail", "nvidia-smi"),
    ]
    for title, cmd in sections:
        print(f"\n--- {title} ---")
        _, out = _run(client, cmd)
        print(out.strip())
    client.close()


# ------------------------------------------------------------------ main

ACTION_MAP = {
    "install": action_install,
    "upload_env": action_upload_env,
    "dry_run": action_dry_run,
    "gpu_run": action_gpu_run,
    "status": action_status,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remote orchestrator for MIRAGE Akash VM"
    )
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument(
        "--action",
        default="all",
        choices=list(ACTION_MAP.keys()) + ["all"],
        help="Action to run (default: all = upload_env → install → dry_run)",
    )
    args = parser.parse_args()

    if args.action == "all":
        action_upload_env(args.host, args.port)
        action_install(args.host, args.port)
        action_dry_run(args.host, args.port)
    else:
        ACTION_MAP[args.action](args.host, args.port)


if __name__ == "__main__":
    main()
