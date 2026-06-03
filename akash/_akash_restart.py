"""
Re-establish Akash VM and resume MIRAGE from last clean checkpoint.

Strategy:
  1. Try to reopen lease on existing DSEQ (preserves /data persistent volume).
  2. If that fails, deploy a new persistent-volume deployment.
  3. Upload .env + repaired code, audit data, start/resume supervisor.
"""
from __future__ import annotations

import json
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

UPLOADS = [
    "parse_utils.py",
    "results_utils.py",
    "Dataset/category_utils.py",
    "GPU_CPU/osm_behavioral.py",
    "GPU_CPU/cdva_patching.py",
    "CPU_Only/scoring.py",
    "CPU_Only/leaderboard.py",
    "CPU_Only/api_behavioral.py",
]

POST_DEPLOY_SH = r"""set -euo pipefail
echo "=== audit /data ==="
ls -la /data/state/ 2>/dev/null || mkdir -p /data/state
/data/venv/bin/python 2>/dev/null <<'PY' || python3 <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
try:
    import pandas as pd
    from results_utils import dedup_behavioral, reparse_failed_rows
    from Dataset.validate_pentad import assert_production_ready
    p = Path("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
    if p.exists():
        df = pd.read_parquet(p)
        assert_production_ready(df)
        print("PENTAD_OK", len(df), df.seed_id.nunique())
    else:
        print("PENTAD_MISSING")
    b = Path("/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet")
    if b.exists():
        raw = pd.read_parquet(b)
        clean = dedup_behavioral(reparse_failed_rows(raw))
        clean.to_parquet(b, index=False)
        print("BEHAVIORAL_OK", len(clean), "fail", int((~clean.success_flag).sum()), "dup", int(clean.duplicated(["prompt_id","model_name","sample_index"]).sum()))
        for m in sorted(clean.model_name.unique()):
            si0 = clean[(clean.model_name==m)&(clean.sample_index==0)]
            print(" ", m, "si0", int(si0.success_flag.sum()), len(si0))
    else:
        print("BEHAVIORAL_MISSING")
except Exception as e:
    print("AUDIT_ERR", e)
PY

echo "=== clear stale GPU markers if pentad valid ==="
if [ -f /data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet ]; then
  /data/venv/bin/python - <<'PY' 2>/dev/null || true
import sys
sys.path.insert(0,"/data/Audit_Benchmark/Code/mirage")
import pandas as pd
from Dataset.validate_pentad import assert_production_ready
df=pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
assert_production_ready(df)
print("pentad gate pass")
PY
  if [ $? -eq 0 ]; then
    rm -f /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
    rm -f /data/state/autonomous_guard.lock
    echo "cleared GPU_PIPELINE_OK PIPELINE_COMPLETE autonomous_guard.lock"
  fi
fi

echo "=== ensure supervisor ==="
pkill -f run_gpu_pipeline.py 2>/dev/null || true
sleep 2
if ! pgrep -f supervise_pipeline.sh >/dev/null; then
  nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
  echo "supervisor started"
else
  echo "supervisor already running"
fi
pgrep -af 'supervise_pipeline|run_gpu_pipeline|_full_pipeline' || true
"""


def _load_key() -> str:
    for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("AKASH_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("AKASH_API_KEY missing in .env")


def _headers(key: str) -> dict:
    return {"x-api-key": key, "Content-Type": "application/json"}


def _parse_ssh_file() -> tuple[str, int, str]:
    host, port, dseq = "", 0, ""
    if VM_SSH.exists():
        for line in VM_SSH.read_text().splitlines():
            if line.startswith("HOST="):
                host = line.split("=", 1)[1].strip()
            elif line.startswith("PORT="):
                port = int(line.split("=", 1)[1].strip() or 0)
            elif line.startswith("DSEQ="):
                dseq = line.split("=", 1)[1].strip()
    return host, port, dseq


def _find_ssh_in_deployment(key: str, dseq: str) -> tuple[str, int]:
    dr = requests.get(f"{BASE}/v1/deployments/{dseq}", headers=_headers(key), timeout=30)
    if dr.status_code != 200:
        return "", 0

    def find_port(obj, depth=0):
        if depth > 8 or not obj:
            return None, None
        if isinstance(obj, dict):
            h = obj.get("host") or obj.get("ip") or obj.get("externalIp") or ""
            p = obj.get("externalPort") or obj.get("port")
            if h and p:
                return h, int(p)
            for v in obj.values():
                res = find_port(v, depth + 1)
                if res[0]:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_port(item, depth + 1)
                if res[0]:
                    return res
        return None, None

    return find_port(dr.json().get("data", {})) or ("", 0)


def _try_ssh(host: str, port: int, timeout: int = 20) -> bool:
    if not host or not port:
        return False
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, port=port, username="root", password=PW, timeout=timeout, banner_timeout=60)
        c.close()
        return True
    except Exception:
        return False


def _try_reopen_existing(key: str, dseq: str) -> tuple[str, int]:
    """Bump deposit and wait for open bid + lease on existing deployment."""
    print(f"[restart] Trying to reopen DSEQ {dseq} (preserve /data) ...")
    r = requests.put(
        f"{BASE}/v1/deployments/{dseq}",
        headers=_headers(key),
        json={"data": {"deposit": 5.0}},
        timeout=30,
    )
    print(f"[restart] PUT deployment: {r.status_code}")

    manifest = ""
    dr = requests.get(f"{BASE}/v1/deployments/{dseq}", headers=_headers(key), timeout=30)
    if dr.status_code == 200:
        manifest = dr.json().get("data", {}).get("deployment", {}).get("hash", "")

    bid = None
    for i in range(1, 19):
        time.sleep(10)
        br = requests.get(f"{BASE}/v1/bids", headers=_headers(key), params={"dseq": dseq}, timeout=30)
        bids = br.json().get("data", [])
        open_bids = [b for b in bids if b.get("bid", b).get("state") == "open"]
        print(f"  t={i*10}s: {len(bids)} bid(s), open={len(open_bids)}")
        if open_bids:
            bid = open_bids[0]
            break

    if not bid:
        host, port = _find_ssh_in_deployment(key, dseq)
        if _try_ssh(host, port):
            print(f"[restart] SSH already up: {host}:{port}")
            return host, port
        return "", 0

    bid_data = bid.get("bid", bid)
    bid_id = bid_data.get("id", {})
    provider = bid_id.get("provider")
    gseq = bid_id.get("gseq", 1)
    oseq = bid_id.get("oseq", 1)

    # Fetch manifest from deployment create response if needed
    mr = requests.get(f"{BASE}/v1/deployments/{dseq}", headers=_headers(key), timeout=30)
    manifest_blob = ""
    if mr.status_code == 200:
        manifest_blob = mr.json().get("data", {}).get("manifest", "") or ""

    lr = requests.post(
        f"{BASE}/v1/leases",
        headers=_headers(key),
        json={
            "manifest": manifest_blob,
            "leases": [{"dseq": dseq, "gseq": gseq, "oseq": oseq, "provider": provider}],
        },
        timeout=60,
    )
    print(f"[restart] Create lease: {lr.status_code} {lr.text[:200]}")

    for j in range(1, 41):
        time.sleep(15)
        host, port = _find_ssh_in_deployment(key, dseq)
        if _try_ssh(host, port):
            print(f"[restart] SSH ready: {host}:{port}")
            return host, port
        if j % 4 == 0:
            print(f"  waiting SSH t={j*15}s ...")
    return "", 0


def _upload_and_resume(host: str, port: int) -> None:
    print(f"[restart] Connecting {host}:{port} ...")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, 15):
        try:
            c.connect(host, port=port, username="root", password=PW, timeout=30, banner_timeout=90)
            break
        except Exception as exc:
            print(f"  SSH attempt {attempt}: {exc}")
            time.sleep(15)
    else:
        raise SystemExit("SSH failed after retries")

    c.exec_command("mkdir -p /data", timeout=10)
    time.sleep(2)
    sftp = c.open_sftp()
    sftp.put(str(ENV_FILE), "/data/.env")
    remote_root = "/data/Audit_Benchmark/Code/mirage"
    for rel in UPLOADS:
        local = REPO / "Code" / "mirage" / rel
        remote = f"{remote_root}/{rel}"
        sftp.put(str(local), remote)
        print(f"  uploaded {rel}")
    sftp.close()
    c.exec_command("chmod 600 /data/.env", timeout=5)

    _, o, e = c.exec_command(f"bash -s << 'EOF'\n{POST_DEPLOY_SH}\nEOF", timeout=180)
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print(err)
    c.close()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    key = _load_key()
    host, port, old_dseq = _parse_ssh_file()

    if old_dseq:
        host, port = _try_reopen_existing(key, old_dseq)

    if not host or not port:
        print("[restart] Reopen failed — running full new deployment ...")
        import subprocess

        r = subprocess.run([sys.executable, str(REPO / "akash" / "_deploy_mirage.py")], cwd=str(REPO))
        if r.returncode != 0:
            raise SystemExit(r.returncode)
        host, port, _ = _parse_ssh_file()
        if not host or not port:
            raise SystemExit("Deploy finished but SSH details missing in akash/vm_ssh.txt")

    _upload_and_resume(host, port)
    VM_SSH.write_text(f"HOST={host}\nPORT={port}\nDSEQ={old_dseq or 'new'}\n")
    print("\n[restart] Done. Monitor with: python akash/_progress_check_now.py")


if __name__ == "__main__":
    main()
