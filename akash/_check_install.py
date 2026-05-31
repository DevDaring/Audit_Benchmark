import paramiko, sys, time

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=20):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode('utf-8', errors='replace')

# Is install still running in tmux?
tmux = run('tmux ls 2>&1')
print("TMUX:", tmux.strip())

# Tail install log
log = run('tail -40 /workspace/install.log 2>/dev/null || echo NO_LOG')
print("\n=== INSTALL LOG (last 40 lines) ===")
print(log)

# Quick package check
pkgs = run('python3 -c "import torch, flash_attn, transformer_lens, nnsight; print(\'ALL_OK\', torch.__version__, flash_attn.__version__)" 2>&1')
print("=== PACKAGES ===", pkgs.strip())

client.close()
