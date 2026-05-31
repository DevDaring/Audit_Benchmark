"""Find the Python that has torch installed and use it everywhere."""
import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('provider.a100.dsm.val.akash.pub', port=30594, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = (o.read() + e.read()).decode('utf-8', 'replace').strip()
    return out

# Find ALL python executables
print("All python executables:")
print(run("find /usr /opt /root /home -name 'python3*' -type f 2>/dev/null | head -20"))

# Check which one has pip
print("\nPython with pip:")
print(run("for py in /usr/bin/python3.10 /usr/local/bin/python3.10 /usr/local/bin/python3 /opt/conda/bin/python3 /opt/python/bin/python3; do [ -f $py ] && $py -m pip --version 2>/dev/null && echo $py; done"))

# Check PATH in interactive shell  
print("\nPATH in bash login shell:")
print(run("bash -l -c 'echo $PATH'"))

# Check PATH when install.sh ran (via tmux)
print("\nPython in tmux PATH:")
print(run("bash -l -c 'which python3; python3 -c \"import torch; print(torch.__version__)\" 2>&1'"))

# Find which python has torch
print("\nPython binaries that have torch:")
print(run("for py in $(find /usr /opt /root -name 'python3*' -type f 2>/dev/null); do $py -c 'import torch; print(\"FOUND\", \"'$py'\"' 2>/dev/null; done"))

c.close()
