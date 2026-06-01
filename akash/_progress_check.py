"""Quick progress check — reads live VM logs and parquet row counts."""
import sys, io, paramiko, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)

def run(cmd, t=30):
    _, o, _ = c.exec_command(cmd, timeout=t)
    return o.read().decode('utf-8', 'replace').strip()

print('=== STATE MARKERS ===')
print(run('ls /data/state/'))

print('\n=== LAST 8 PROGRESS LINES ===')
print(run('tail -8 /data/logs/pipeline_attempt_1.log 2>/dev/null'))

print('\n=== RESULTS FILES ===')
print(run('ls -lh /data/Audit_Benchmark/Code/mirage/results/ 2>/dev/null || echo empty'))

print('\n=== PARQUET ROW COUNTS ===')
py_script = """
import os, sys
r = '/data/Audit_Benchmark/Code/mirage/results'
for fname in ['behavioral_results.parquet', 'cdva_results.parquet']:
    p = os.path.join(r, fname)
    if os.path.exists(p):
        try:
            import pandas as pd
            df = pd.read_parquet(p)
            ok = df['success_flag'].sum() if 'success_flag' in df.columns else '?'
            print(f'{fname}: {len(df)} rows, {ok} successful')
        except Exception as e:
            print(f'{fname}: CORRUPT? {e}')
    else:
        print(f'{fname}: not yet written')
"""
print(run(f'/data/venv/bin/python -c "{py_script}"', t=30))

print('\n=== GPU VRAM + UTILISATION ===')
print(run('nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader'))

print('\n=== CONTAINER AGE + DISK ===')
print('PID1 age (s):', run('ps -p 1 -o etimes= | tr -d " "'))
print('Disk /data:', run('df -h /data | tail -1'))

print('\n=== RUNNING PROCS ===')
print(run('ps aux --no-header | grep -E "run_gpu_pipeline|behavioral|cdva|_full_pipe" | grep -v grep || echo none'))

c.close()
