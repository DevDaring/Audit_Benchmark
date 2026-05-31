import paramiko, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('provider.a100.dsm.val.akash.pub', port=30594, username='root', password='MirageVM2026!', timeout=30)

def run(cmd):
    _, o, e = c.exec_command(cmd, timeout=25)
    return (o.read() + e.read()).decode('utf-8', 'replace').strip()

print("TMUX    :", run("tmux ls 2>&1"))
print("LOG_LINES:", run("wc -l /workspace/install.log 2>/dev/null || echo 0"))
print("DONE    :", run("grep -c INSTALL_DONE /workspace/install.log 2>/dev/null || echo 0"))
print()
print("=== LAST 10 LOG LINES ===")
print(run("tail -10 /workspace/install.log 2>/dev/null | cat"))
print()
print("=== TORCH ===")
print(run("python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1"))
print("=== FLASH_ATTN ===")
print(run("python3 -c 'import flash_attn; print(flash_attn.__version__)' 2>&1"))

c.close()
