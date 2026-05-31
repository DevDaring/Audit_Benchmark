"""Quick VM package check with proper quoting."""
import paramiko, sys

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

# Use a Python script file to avoid shell quoting issues.
CHECK_SCRIPT = r"""
import sys, importlib
pkgs = [
    "torch", "dotenv", "flash_attn", "transformers",
    "transformer_lens", "nnsight", "outlines",
    "pandas", "pyarrow", "openai", "boto3", "scipy",
]
for p in pkgs:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", "ok")
        print(f"  OK   {p:<24} {v}")
    except Exception as ex:
        print(f"  FAIL {p:<24} {ex}")
import torch
print(f"  CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
"""

# Write the script to VM, then run it.
run(c, "cat > /tmp/check_pkgs.py << 'PYEOF'\n" + CHECK_SCRIPT + "\nPYEOF", t=10)
sp(run(c, "python3 /tmp/check_pkgs.py 2>&1", t=30))

# Also check WHERE packages are installed.
sp("\n--- pip list (relevant) ---")
sp(run(c, "python3 -m pip list 2>/dev/null | grep -E 'dotenv|transformer.lens|nnsight|outlines|torch|flash'", t=20))

# Check sys.path
sp("\n--- sys.path ---")
sp(run(c, "python3 -c \"import sys; [print(p) for p in sys.path]\"", t=10))

c.close()
