"""
Run install.sh inside a tmux session on the VM, then tail the log.
Install takes 5-10 min; tmux ensures it doesn't die on SSH disconnect.
"""
import paramiko, time, sys

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=15):
    _, out, err = client.exec_command(cmd, timeout=timeout)
    o = out.read().decode()
    e = err.read().decode()
    return o + e

# Kill any existing install tmux session
run('tmux kill-session -t install 2>/dev/null || true')
time.sleep(1)

# Start install.sh inside tmux
print("[tmux] Launching install.sh in tmux session 'install' ...")
run('tmux new-session -d -s install -x 250 -y 50')
run('tmux send-keys -t install "bash /workspace/Audit_Benchmark/akash/install.sh 2>&1 | tee /workspace/install.log; echo INSTALL_DONE >> /workspace/install.log" Enter')
print("[tmux] install.sh started. Tailing /workspace/install.log ...")
print("[tmux] This will stream for ~8 minutes.")
print()

# Stream the log file with ssh exec in a loop
for i in range(1, 100):
    time.sleep(10)
    log_tail = run('tail -15 /workspace/install.log 2>/dev/null || echo "starting..."')
    print("--- t=%ds ---" % (i * 10))
    print(log_tail)
    
    # Check for completion
    if 'INSTALL_DONE' in log_tail or 'DONE' in log_tail:
        print("\n[tmux] install.sh COMPLETED.")
        break
    if 'ERROR' in log_tail and 'flash' in log_tail.lower():
        print("\n[tmux] flash-attn install ERROR. Check log.")
        break

# Final verification
print("\n=== FINAL PACKAGE CHECK ===")
print(run('python3 -c "import torch, flash_attn, transformer_lens, nnsight; print(\'ALL OK\', torch.__version__, flash_attn.__version__, transformer_lens.__version__)" 2>&1'))

client.close()
