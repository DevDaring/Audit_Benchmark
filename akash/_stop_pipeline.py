"""Stop corrupt GPU run and check for local DeepSeek checkpoints on VM."""
import sys, io, paramiko
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)

cmds = [
    'pkill -f run_gpu_pipeline.py; pkill -f _full_pipeline.py; echo STOPPED',
    'ls -la /data/Audit_Benchmark/Code/mirage/Dataset/seeds/',
    'ls -la /data/Audit_Benchmark/Code/mirage/Dataset/seeds/*.json 2>/dev/null || echo no checkpoints',
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, timeout=20)
    print(f'--- {cmd} ---')
    print(o.read().decode('utf-8','replace'))
    err = e.read().decode('utf-8','replace')
    if err.strip(): print('err:', err[:200])
c.close()
