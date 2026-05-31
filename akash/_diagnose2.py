"""Deep diagnostics: container uptime, PID 1, storage, memory."""
import paramiko

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

c = conn()
sp("=== PID 1 uptime ===")
sp(run(c, "ps -p 1 -o etimes,comm,args"))

sp("\n=== All processes ===")
sp(run(c, "ps aux --no-header | head -15"))

sp("\n=== Memory ===")
sp(run(c, "free -h"))

sp("\n=== Storage ===")
sp(run(c, "df -h"))

sp("\n=== /workspace contents ===")
sp(run(c, "ls -la /workspace/"))

sp("\n=== /usr/local/lib/python3.10 ===")
sp(run(c, "ls /usr/local/lib/python3.10/ 2>/dev/null | head -20"))

sp("\n=== dist-packages contents (first 30) ===")
sp(run(c, "ls /usr/local/lib/python3.10/dist-packages/ 2>/dev/null | head -30 || echo EMPTY"))

sp("\n=== pip list ===")
sp(run(c, "python3 -m pip list 2>/dev/null | head -30 || echo NO_PIP"))
c.close()
