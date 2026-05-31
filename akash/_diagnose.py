"""Diagnose Python path split and force-install dotenv."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('provider.a100.dsm.val.akash.pub', port=30594, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = (o.read() + e.read()).decode('utf-8', 'replace').strip()
    rc = o.channel.recv_exit_status()
    return rc, out

# Find all python3 binaries
_, out = run("which python3; python3 --version; which python3.10 2>/dev/null; ls /usr/local/bin/python* 2>/dev/null; ls /usr/bin/python* 2>/dev/null")
print("PYTHON BINARIES:\n", out)

# Check sys.path
_, out = run("python3 -c 'import sys; print(\"\\n\".join(sys.path))'")
print("\nSYS.PATH:\n", out)

# Check where pip installs
_, out = run("python3 -m pip show pip 2>&1 | head -5")
print("\nPIP LOCATION:\n", out)

# Where is torch installed?
_, out = run("python3 -c 'import torch; print(torch.__file__)' 2>&1")
print("\nTORCH FILE:\n", out)

# Is dotenv installed at all?
_, out = run("python3 -m pip list 2>&1 | grep -i dotenv")
print("\nDOTENV IN PIP LIST:\n", out or "(not found)")

# Force install verbose
print("\n=== FORCE INSTALL python-dotenv verbose ===")
_, out = run("python3 -m pip install --force-reinstall python-dotenv 2>&1 | tail -15", timeout=60)
print(out)

# Try importing after force install
_, out = run("python3 -c 'from dotenv import load_dotenv; print(\"DOTENV OK\")'")
print("\nIMPORT TEST:", out)

# Also install outlines
print("\n=== INSTALL outlines ===")
_, out = run("python3 -m pip install outlines 2>&1 | tail -5", timeout=120)
print(out)

c.close()
