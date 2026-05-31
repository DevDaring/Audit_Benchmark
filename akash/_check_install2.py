import paramiko, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=20):
    _, o, e = client.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode('utf-8', 'replace')

print("TMUX:", run("tmux ls 2>&1").strip())
print()

log = run("tail -35 /workspace/install.log 2>/dev/null | cat")
print("=== INSTALL LOG ===")
print(log)

pkgs = run("python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())' 2>&1")
print("=== torch ===", pkgs.strip())

fa = run("python3 -c 'import flash_attn; print(flash_attn.__version__)' 2>&1")
print("=== flash_attn ===", fa.strip())

done = run("grep -c INSTALL_DONE /workspace/install.log 2>/dev/null || echo 0")
print("=== INSTALL_DONE count ===", done.strip())

client.close()
