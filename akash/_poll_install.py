"""Poll VM every 30s until install.sh completes."""
import paramiko, sys, io, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, t=20):
    _, o, e = client.exec_command(cmd, timeout=t)
    return (o.read() + e.read()).decode('utf-8', 'replace')

for i in range(1, 40):
    time.sleep(30)
    log = run("tail -5 /workspace/install.log 2>/dev/null | cat")
    done = "1" in run("grep -c INSTALL_DONE /workspace/install.log 2>/dev/null")
    print("t=%dm | %s" % (i // 2, log.strip().splitlines()[-1] if log.strip() else "no log"))
    if done:
        print("\nINSTALL COMPLETE!")
        break
    if "ERROR" in log or "error" in log.lower():
        full = run("grep -i 'error\\|failed' /workspace/install.log 2>/dev/null | tail -10 | cat")
        print("POSSIBLE ERROR:\n", full)

# Final check
r = run("python3 -c 'import torch, flash_attn, transformer_lens; print(torch.__version__, flash_attn.__version__, transformer_lens.__version__)' 2>&1")
print("\nFINAL:", r.strip())
client.close()
