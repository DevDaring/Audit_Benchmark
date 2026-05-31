"""
Install all packages AND run 2-seed dry run in one chained tmux command.

This avoids the race condition where: install completes -> container restarts
-> packages are gone before dry run starts.

By chaining with && inside a single tmux session, the dry run starts
immediately when install finishes — no SSH disconnect, no gap.
"""
import paramiko, sys, time

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 30594
LOG = "/workspace/install_and_dry.log"
POLL = 30
MAX_MIN = 90   # install ~30 min + dry run ~20 min


def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
              timeout=30, banner_timeout=60)
    return c


def run(c, cmd, t=20):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


def sp(s):
    try: print(s)
    except UnicodeEncodeError: print(s.encode("ascii", errors="replace").decode())


# ---- Pull latest code ----
c = conn()
sp("[1] git pull ...")
sp(run(c, "cd /workspace/Audit_Benchmark && git pull 2>&1", t=60)[:400])

# ---- Kill any stale sessions ----
run(c, "tmux kill-session -t pipeline 2>/dev/null; true")

# ---- Build the single chained command ----
INSTALL = "bash /workspace/Audit_Benchmark/akash/install.sh"
DRY_RUN = (
    "cd /workspace/Audit_Benchmark/Code/mirage && "
    "PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage "
    "python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2"
)
CHAIN = (
    f"{INSTALL} 2>&1 | tee {LOG}"
    f" && echo INSTALL_OK >> {LOG}"
    f" && {DRY_RUN} 2>&1 | tee -a {LOG}"
    f"; echo PIPELINE_DONE >> {LOG}"
)
tmux_cmd = f"tmux new-session -d -s pipeline '{CHAIN}'"
sp(f"\n[2] Starting install + dry run pipeline in tmux ...")
sp(run(c, tmux_cmd, t=15))
time.sleep(2)
sp(run(c, "tmux list-sessions 2>&1", t=5))
c.close()
sp(f"\n  Log: {LOG}")
sp("  Polling every 30s ...\n")

# ---- Poll ----
deadline = time.time() + MAX_MIN * 60
polls = 0
install_done = False
while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c2 = conn()
        done = run(c2, f"grep -c PIPELINE_DONE {LOG} 2>/dev/null || echo 0")
        inst = run(c2, f"grep -c INSTALL_OK {LOG} 2>/dev/null || echo 0")
        tail = run(c2, f"tail -6 {LOG}")
        c2.close()
    except Exception as ex:
        sp(f"  [{polls*POLL}s] reconnect: {ex}")
        continue

    if not install_done and inst.strip().endswith("1"):
        sp(f"\n  *** INSTALL COMPLETE at {polls*POLL}s — dry run starting ***")
        install_done = True

    sp(f"[{polls*POLL}s] INST={'Y' if install_done else 'N'} DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        sp("  " + ln)

    if done.strip().endswith("1"):
        break
else:
    sp("TIMEOUT.")
    sys.exit(1)

# ---- Final result ----
c = conn()
sp("\n" + "=" * 60)
sp("FINAL LOG TAIL (last 80 lines):")
sp("=" * 60)
sp(run(c, f"tail -80 {LOG}", t=60))

fails = int(run(c, f"grep -c ' FAIL' {LOG} 2>/dev/null || echo 0").split()[-1])
passes = int(run(c, f"grep -c ' PASS' {LOG} 2>/dev/null || echo 0").split()[-1])
c.close()

sp(f"\nRESULT: {passes} PASS lines, {fails} FAIL lines in log.")
if fails == 0:
    sp("PIPELINE PASSED.")
else:
    sp("Pipeline had FAILs. Review log above.")
    sys.exit(1)
