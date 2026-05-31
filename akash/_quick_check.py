"""Quick check: is the VM reachable? What's running?"""
import paramiko, sys

def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("provider.a100.dsm.val.akash.pub", port=30594, username="root",
              password="MirageVM2026!", timeout=15, banner_timeout=30)
    return c

def run(c, cmd, t=15):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()

try:
    c = conn()
    print("CONNECTED")
    print("tmux:", run(c, "tmux list-sessions 2>&1"))
    print("uptime:", run(c, "uptime"))
    print("workspace:", run(c, "ls /workspace/"))
    print("log exists:", run(c, "ls -la /workspace/install_and_dry.log 2>/dev/null || echo NO_LOG"))
    print("log tail:", run(c, "tail -5 /workspace/install_and_dry.log 2>/dev/null || echo EMPTY"))
    c.close()
except Exception as ex:
    print(f"FAILED: {ex}")
    sys.exit(1)
