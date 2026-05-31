"""
Upload .env to the VM via SFTP and immediately launch the 2-seed dry run
in tmux (packages are already installed from the previous install run).
"""
import paramiko, sys, time
from pathlib import Path

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 30594
LOCAL_ENV = Path(__file__).resolve().parents[1] / "Code" / "mirage" / ".env"
REMOTE_ENV = "/workspace/Audit_Benchmark/Code/mirage/.env"
DRY_LOG = "/workspace/dry_run_2seeds.log"
POLL = 20
MAX_MIN = 30


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


# ---- 1. Upload .env ----
sp(f"[1] Uploading .env ({LOCAL_ENV.stat().st_size} bytes) ...")
c = conn()
sftp = c.open_sftp()
sftp.put(str(LOCAL_ENV), REMOTE_ENV)
sftp.close()
sp(f"    Uploaded to {REMOTE_ENV}")

# Verify HUGGINGFACE_TOKEN is present (don't print value).
hf_check = run(c, f"grep -c HUGGINGFACE_TOKEN {REMOTE_ENV} 2>/dev/null || echo 0")
sp(f"    HUGGINGFACE_TOKEN lines in .env: {hf_check.split()[-1]}")

# ---- 2. Quick package sanity (packages should still be installed) ----
sp("\n[2] Quick package check ...")
torch_ok = run(c, "python3 -c \"import torch; print(torch.__version__, torch.cuda.is_available())\" 2>&1", t=20)
dot_ok   = run(c, "python3 -c \"import dotenv; print(dotenv.__version__)\" 2>&1", t=10)
sp(f"    torch: {torch_ok[:60]}")
sp(f"    dotenv: {dot_ok[:40]}")
if "Traceback" in torch_ok or "Traceback" in dot_ok:
    sp("    ERROR: packages not available. Re-run _install_and_dryrun.py first.")
    c.close()
    sys.exit(1)

# ---- 3. Launch dry run in tmux ----
run(c, "tmux kill-session -t dry2 2>/dev/null; true")
run_cmd = (
    "cd /workspace/Audit_Benchmark/Code/mirage && "
    "PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage "
    "python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2"
    " 2>&1 | tee " + DRY_LOG +
    "; echo DRY_RUN_DONE >> " + DRY_LOG
)
tmux_cmd = "tmux new-session -d -s dry2 '" + run_cmd + "'"
sp("\n[3] Launching 2-seed dry run in tmux ...")
sp(run(c, tmux_cmd, t=10))
time.sleep(2)
sp(run(c, "tmux list-sessions 2>&1", t=5))
c.close()
sp(f"    Log: {DRY_LOG}\n")

# ---- 4. Poll ----
deadline = time.time() + MAX_MIN * 60
polls = 0
while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c2 = conn()
        done = run(c2, f"grep -c DRY_RUN_DONE {DRY_LOG} 2>/dev/null || echo 0")
        tail = run(c2, f"tail -8 {DRY_LOG}")
        c2.close()
    except Exception as ex:
        sp(f"[{polls*POLL}s] reconnect: {ex}"); continue
    sp(f"[{polls*POLL}s] DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        sp("  " + ln)
    if done.strip().endswith("1"):
        break
else:
    sp("TIMEOUT."); sys.exit(1)

# ---- 5. Final result ----
c = conn()
sp("\n" + "=" * 60)
sp("FULL DRY RUN LOG:")
sp("=" * 60)
sp(run(c, f"cat {DRY_LOG}", t=60))
fails  = int(run(c, f"grep -c ' FAIL' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
passes = int(run(c, f"grep -c ' PASS' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
c.close()
sp(f"\nRESULT: {passes} PASS, {fails} FAIL")
if fails == 0:
    sp("2-SEED DRY RUN PASSED.")
else:
    sp("Dry run FAILs found.")
    sys.exit(1)
