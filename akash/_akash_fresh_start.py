"""
Fresh Akash deployment per official lifecycle docs.

Akash facts (docs + marketplace spec):
  - A CLOSED lease is terminal — you cannot SSH or restart it.
  - Persistent volumes survive container RESTARTS within one active lease only.
  - Persistent data is NOT kept across closed leases (even same provider).
  - Recovery = new deployment → wait for open bids → POST /v1/leases with manifest.

This script:
  1. Closes stale deployments (recover escrow).
  2. Creates a new persistent-volume deployment (_deploy_mirage.py SDL).
  3. Uploads .env + repaired MIRAGE code (supervisor uses MIRAGE_GIT_PULL=0).
  4. Clears GPU-only markers on fresh /data; leaves install/dataset to pipeline.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import paramiko
import requests

REPO = Path(__file__).resolve().parents[1]
ENV_FILE = REPO / "Code" / "mirage" / ".env"
VM_SSH = REPO / "akash" / "vm_ssh.txt"
BASE = "https://console-api.akash.network"
PW = "MirageVM2026!"

CLOSE_DSEQS = ["27072516", "27070590", "27070564", "27070538", "27070733", "27071620"]

UPLOADS = [
    "parse_utils.py",
    "results_utils.py",
    "Dataset/category_utils.py",
    "GPU_CPU/osm_behavioral.py",
    "GPU_CPU/cdva_patching.py",
    "CPU_Only/scoring.py",
    "CPU_Only/leaderboard.py",
    "CPU_Only/api_behavioral.py",
    "GPU_CPU/run_gpu_pipeline.py",
    "GPU_CPU/pipeline_guards.py",
    "Dataset/validate_pentad.py",
]


def _load_key() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AKASH_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("AKASH_API_KEY missing")


def _headers(key: str) -> dict:
    return {"x-api-key": key, "Content-Type": "application/json"}


def close_old_deployments(key: str) -> None:
    print("[fresh] Closing stale deployments (recover escrow) ...")
    for dseq in CLOSE_DSEQS:
        r = requests.delete(f"{BASE}/v1/deployments/{dseq}", headers=_headers(key), timeout=30)
        if r.status_code in (200, 201, 204):
            print(f"  closed DSEQ {dseq}")
        else:
            print(f"  DSEQ {dseq}: {r.status_code} (may already be closed)")


def deploy_new() -> tuple[str, int]:
    print("[fresh] Creating new deployment via _deploy_mirage.py ...")
    r = subprocess.run(
        [sys.executable, str(REPO / "akash" / "_deploy_mirage.py")],
        cwd=str(REPO),
    )
    if r.returncode != 0:
        raise SystemExit("Deploy script failed")
    host, port, dseq = "", 0, ""
    for line in VM_SSH.read_text().splitlines():
        if line.startswith("HOST="):
            host = line.split("=", 1)[1].strip()
        elif line.startswith("PORT="):
            port = int(line.split("=", 1)[1].strip() or 0)
        elif line.startswith("DSEQ="):
            dseq = line.split("=", 1)[1].strip()
    if not host or not port:
        raise SystemExit("SSH details missing after deploy")
    return host, port, dseq


def upload_code_and_env(host: str, port: int) -> None:
    print(f"[fresh] Uploading .env + repaired code to {host}:{port} ...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, 20):
        try:
            c.connect(host, port=port, username="root", password=PW, timeout=30, banner_timeout=90)
            break
        except Exception as exc:
            print(f"  SSH attempt {attempt}: {exc}")
            time.sleep(20)
    else:
        raise SystemExit("SSH never became available")

    c.exec_command("mkdir -p /data/state /data/logs", timeout=10)
    time.sleep(2)
    sftp = c.open_sftp()
    sftp.put(str(ENV_FILE), "/data/.env")
    remote_root = "/data/Audit_Benchmark/Code/mirage"
    for rel in UPLOADS:
        local = REPO / "Code" / "mirage" / rel
        if local.exists():
            sftp.put(str(local), f"{remote_root}/{rel}")
            print(f"  uploaded {rel}")
    sftp.close()
    c.exec_command("chmod 600 /data/.env", timeout=5)

    # Fresh lease: no stale GPU markers; supervisor will run full pipeline.
    setup = r"""
set -euo pipefail
rm -f /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
rm -f /data/state/autonomous_guard.lock
# Do NOT touch INSTALL_OK / PREDOWNLOAD_OK / DATASET_OK on fresh volume they won't exist yet.
if ! pgrep -f supervise_pipeline.sh >/dev/null 2>&1; then
  nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
fi
echo "SUPERVISOR=$(pgrep -af supervise_pipeline || echo none)"
ls -la /data/state/ 2>/dev/null || true
"""
    _, o, e = c.exec_command(f"bash -s << 'EOF'\n{setup}\nEOF", timeout=60)
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print(err)
    c.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = _load_key()
    close_old_deployments(key)
    host, port, dseq = deploy_new()
    upload_code_and_env(host, port)
    VM_SSH.write_text(f"HOST={host}\nPORT={port}\nDSEQ={dseq}\n")
    print()
    print("=" * 70)
    print("  FRESH AKASH DEPLOYMENT STARTED")
    print("=" * 70)
    print(f"  DSEQ:     {dseq}")
    print(f"  SSH:      ssh root@{host} -p {port}")
    print(f"  Password: {PW}")
    print()
    print("  Previous lease data is NOT recoverable (Akash: closed lease = terminal).")
    print("  Full clean run: install → predownload → pentad → GPU (~15–24 h wall).")
    print("  Monitor: python akash/_progress_check_now.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
