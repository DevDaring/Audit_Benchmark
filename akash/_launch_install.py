"""Launch install.sh in a tmux session on the VM, then poll until done."""
import paramiko, time, sys

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 30594
VM_USER = "root"
VM_PASS = "MirageVM2026!"
LOG = "/workspace/reinstall.log"
POLL_INTERVAL = 30
MAX_WAIT_MIN = 60


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username=VM_USER, password=VM_PASS, timeout=30, banner_timeout=60)
    return c


def run(c, cmd, timeout=30):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


client = connect()

# Pull latest code.
print("[1] git pull ...")
print(run(client, "cd /workspace/Audit_Benchmark && git pull 2>&1", timeout=60)[:400])

# Kill stale sessions.
run(client, "tmux kill-session -t reinstall 2>/dev/null; true")

# Start fresh install.
tmux_cmd = (
    "tmux new-session -d -s reinstall "
    + "'bash /workspace/Audit_Benchmark/akash/install.sh"
    + " 2>&1 | tee " + LOG
    + "; echo REINSTALL_DONE >> " + LOG + "'"
)
print("[2] Launching tmux reinstall session ...")
print(run(client, tmux_cmd, timeout=15))
time.sleep(2)
print("TMUX sessions:", run(client, "tmux list-sessions 2>&1", timeout=10))
client.close()
print("Install started. Polling...\n")

# Poll loop - reconnect each time.
deadline = time.time() + MAX_WAIT_MIN * 60
polls = 0
while time.time() < deadline:
    time.sleep(POLL_INTERVAL)
    polls += 1
    try:
        c2 = connect()
        done = run(c2, "grep -c REINSTALL_DONE " + LOG + " 2>/dev/null || echo 0", timeout=10)
        tail = run(c2, "tail -6 " + LOG, timeout=15)
        c2.close()
    except Exception as ex:
        print(f"  [{polls * POLL_INTERVAL}s] reconnect error: {ex}")
        continue
    print(f"[{polls * POLL_INTERVAL}s] DONE={done.split()[-1]}")
    for line in tail.splitlines():
        try:
            print("  " + line)
        except UnicodeEncodeError:
            print("  " + line.encode("ascii", errors="replace").decode())
    if done.strip().endswith("1"):
        break
else:
    print("TIMEOUT")
    sys.exit(1)

# Final verification.
client = connect()
print("\n[3] Last 50 log lines:")
print(run(client, "tail -50 " + LOG, timeout=30))
print("\n[4] torch check:")
print(run(client, "python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'", timeout=30))
print("\n[5] dotenv check:")
print(run(client, "python3 -c 'import dotenv; print(\"OK\", dotenv.__version__)'", timeout=15))
client.close()
print("\nDone.")
