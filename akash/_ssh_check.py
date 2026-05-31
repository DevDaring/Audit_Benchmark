"""Check VM install log and GPU status via SSH."""
import paramiko, time

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

# Check install log
_, out, _ = client.exec_command('tail -80 /workspace/install.log 2>/dev/null || echo "no log yet"')
print("=== INSTALL LOG (last 80 lines) ===")
print(out.read().decode())

# Check GPU
_, out2, _ = client.exec_command('nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>&1')
print("\n=== GPU ===")
print(out2.read().decode())

# Check python packages
_, out3, _ = client.exec_command('python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1')
print("\n=== torch ===")
print(out3.read().decode())

_, out4, _ = client.exec_command('python3 -c "import flash_attn; print(flash_attn.__version__)" 2>&1')
print("\n=== flash_attn ===")
print(out4.read().decode())

# Repo status
_, out5, _ = client.exec_command('ls /workspace/Audit_Benchmark/Code/mirage/ 2>&1')
print("\n=== repo ===")
print(out5.read().decode())

client.close()
