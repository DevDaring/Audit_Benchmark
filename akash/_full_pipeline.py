"""
Full pipeline:
  1. Upload .env to /workspace/mirage.env (before install, survives the chain)
  2. Launch tmux: install.sh → cp .env → dry_run_gpu_cpu.py --n-seeds 2
  3. Poll until DRY_RUN_DONE
"""
import paramiko, sys, time
from pathlib import Path

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 32355
LOCAL_ENV = Path(__file__).resolve().parents[1] / "Code" / "mirage" / ".env"
REMOTE_ENV_STAGE = "/workspace/mirage.env"          # staging path
REMOTE_ENV_FINAL = "/workspace/Audit_Benchmark/Code/mirage/.env"  # runtime path
DRY_LOG = "/workspace/full_pipeline.log"
POLL = 30
MAX_MIN = 60


def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
              timeout=15, banner_timeout=30)
    return c


def run(c, cmd, t=15):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


def sp(s):
    try: print(s, flush=True)
    except UnicodeEncodeError: print(s.encode("ascii", errors="replace").decode(), flush=True)


# ---- 1. Upload .env to staging path ----
sp(f"[1] Uploading .env to {REMOTE_ENV_STAGE} ...")
c = conn()
sftp = c.open_sftp()
sftp.put(str(LOCAL_ENV), REMOTE_ENV_STAGE)
sftp.close()
sp(f"    Uploaded ({LOCAL_ENV.stat().st_size} bytes)")
hf = run(c, f"grep -c HUGGINGFACE_TOKEN {REMOTE_ENV_STAGE}")
sp(f"    HF token lines: {hf.split()[-1]}")

# ---- 2. Kill stale sessions ----
run(c, "tmux kill-session -t full 2>/dev/null; true")

# ---- 3. Build chained command ----
# Steps in tmux:
#   a) Run install.sh (all packages)
#   b) Copy .env from staging to runtime path
#   c) Run 2-seed dry run
INSTALL = f"bash /workspace/Audit_Benchmark/akash/install.sh"
COPY_ENV = f"cp {REMOTE_ENV_STAGE} {REMOTE_ENV_FINAL}"
DRY = (
    "cd /workspace/Audit_Benchmark/Code/mirage && "
    "PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage "
    "python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2"
)
CHAIN = (
    f"{INSTALL} 2>&1 | tee {DRY_LOG}"
    f" && echo INSTALL_OK >> {DRY_LOG}"
    f" && {COPY_ENV}"
    f" && {DRY} 2>&1 | tee -a {DRY_LOG}"
    f"; echo PIPELINE_DONE >> {DRY_LOG}"
)
tmux_cmd = f"tmux new-session -d -s full '{CHAIN}'"
sp(f"\n[2] Launching chained pipeline in tmux ...")
sp(run(c, tmux_cmd, t=10))
time.sleep(2)
sp(run(c, "tmux list-sessions 2>&1"))
c.close()
sp(f"    Log: {DRY_LOG}\n")

# ---- 4. Poll ----
deadline = time.time() + MAX_MIN * 60
polls = 0
install_ok = False

while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c2 = conn()
        done  = run(c2, f"grep -c PIPELINE_DONE {DRY_LOG} 2>/dev/null || echo 0")
        iok   = run(c2, f"grep -c INSTALL_OK {DRY_LOG} 2>/dev/null || echo 0")
        size  = run(c2, f"wc -c < {DRY_LOG} 2>/dev/null || echo 0")
        tail  = run(c2, f"tail -8 {DRY_LOG}")
        c2.close()
    except Exception as ex:
        sp(f"[{polls*POLL}s] reconnect: {ex}"); continue

    if not install_ok and iok.strip().endswith("1"):
        sp(f"  *** INSTALL_OK at {polls*POLL}s — .env copied, dry run starting ***")
        install_ok = True

    sp(f"[{polls*POLL}s] log={size.split()[-1]}B INST={'Y' if install_ok else 'N'} DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        sp("  " + ln)

    if done.strip().endswith("1"):
        break
else:
    sp("TIMEOUT"); sys.exit(1)

# ---- 5. Final result ----
c = conn()
sp("\n" + "=" * 60)
sp("FINAL LOG (last 120 lines):")
sp("=" * 60)
sp(run(c, f"tail -120 {DRY_LOG}", t=60))
fails  = int(run(c, f"grep -c ' FAIL' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
passes = int(run(c, f"grep -c ' PASS' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
c.close()
sp(f"\nRESULT: {passes} PASS, {fails} FAIL")
if fails == 0:
    sp("2-SEED DRY RUN PASSED.")
else:
    sp("Dry run FAILs found. Review log above.")
    sys.exit(1)

