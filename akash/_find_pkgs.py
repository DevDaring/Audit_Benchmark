"""Find where torch and packages actually live on the VM."""
import paramiko

def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect("provider.a100.dsm.val.akash.pub", port=30594, username="root",
              password="MirageVM2026!", timeout=30, banner_timeout=60)
    return c

def run(c, cmd, t=20):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()

def sp(s):
    try: print(s)
    except UnicodeEncodeError: print(s.encode("ascii", errors="replace").decode())

c = conn()

sp("--- find torch on filesystem ---")
sp(run(c, "find / -name 'torch' -type d 2>/dev/null | grep -v proc | head -10", t=15))

sp("\n--- find pip executables ---")
sp(run(c, "which python3; which pip3; ls -la /usr/local/bin/pip* /usr/bin/pip* 2>/dev/null", t=10))

sp("\n--- python3 --version and path ---")
sp(run(c, "which python3; python3 --version", t=5))

sp("\n--- pip3 list (first 20) ---")
sp(run(c, "pip3 list 2>/dev/null | head -20 || python3 -m pip list 2>/dev/null | head -20 || echo no_pip", t=15))

sp("\n--- /usr/local/lib python dirs ---")
sp(run(c, "ls /usr/local/lib/ 2>/dev/null", t=5))

sp("\n--- /root/.local dist-packages ---")
sp(run(c, "ls /root/.local/lib/ 2>/dev/null", t=5))

c.close()
