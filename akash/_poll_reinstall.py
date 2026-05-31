"""Poll the running reinstall on the VM (reconnects each time)."""
import paramiko, sys, time

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 30594
LOG = "/workspace/reinstall.log"
POLL = 30
MAX_MIN = 60


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!", timeout=30, banner_timeout=60)
    return c


def run(c, cmd, timeout=20):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


def safe_print(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode())


deadline = time.time() + MAX_MIN * 60
polls = 0
while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c = connect()
        done = run(c, f"grep -c REINSTALL_DONE {LOG} 2>/dev/null || echo 0")
        tail = run(c, f"tail -8 {LOG}")
        c.close()
    except Exception as ex:
        safe_print(f"[{polls*POLL}s] reconnect error: {ex}")
        continue

    safe_print(f"\n[{polls*POLL}s] DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        safe_print("  " + ln)

    if done.strip().endswith("1"):
        print("\nInstall DONE.")
        break
else:
    print("TIMEOUT"); sys.exit(1)

# Verify.
c = connect()
print("\n--- Last 50 lines ---")
safe_print(run(c, f"tail -50 {LOG}", timeout=30))
print("\n--- torch ---")
safe_print(run(c, "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'", timeout=30))
print("\n--- dotenv ---")
safe_print(run(c, "python3 -c 'import dotenv; print(dotenv.__version__)'", timeout=15))
c.close()
