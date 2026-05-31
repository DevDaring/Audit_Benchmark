"""Poll /workspace/install_and_dry.log until PIPELINE_DONE appears."""
import paramiko, sys, time

def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("provider.a100.dsm.val.akash.pub", port=30594, username="root",
              password="MirageVM2026!", timeout=15, banner_timeout=30)
    return c

def run(c, cmd, t=15):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()

def sp(s):
    try: print(s, flush=True)
    except UnicodeEncodeError: print(s.encode("ascii", errors="replace").decode(), flush=True)

LOG = "/workspace/install_and_dry.log"
POLL = 30
MAX_MIN = 90

deadline = time.time() + MAX_MIN * 60
polls = 0
install_ok = False

while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c = conn()
        done  = run(c, f"grep -c PIPELINE_DONE {LOG} 2>/dev/null || echo 0")
        iok   = run(c, f"grep -c INSTALL_OK {LOG} 2>/dev/null || echo 0")
        size  = run(c, f"wc -c < {LOG} 2>/dev/null || echo 0")
        tail  = run(c, f"tail -8 {LOG}")
        c.close()
    except Exception as ex:
        sp(f"[{polls*POLL}s] reconnect err: {ex}")
        continue

    if not install_ok and iok.strip().endswith("1"):
        sp(f"  *** INSTALL_OK at {polls*POLL}s — dry run is starting ***")
        install_ok = True

    sp(f"[{polls*POLL}s] log={size.split()[-1]}B INST={'Y' if install_ok else 'N'} DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        sp("  " + ln)

    if done.strip().endswith("1"):
        break
else:
    sp("TIMEOUT"); sys.exit(1)

# Final result.
c = conn()
sp("\n=== FINAL LOG (last 100 lines) ===")
sp(run(c, f"tail -100 {LOG}", t=30))
fails  = int(run(c, f"grep -c ' FAIL' {LOG} 2>/dev/null || echo 0").split()[-1])
passes = int(run(c, f"grep -c ' PASS' {LOG} 2>/dev/null || echo 0").split()[-1])
c.close()
sp(f"\nRESULT: {passes} PASS, {fails} FAIL")
sp("PIPELINE PASSED." if fails == 0 else "Pipeline FAILs found.")
