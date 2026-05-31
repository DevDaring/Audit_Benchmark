"""Quick VM state snapshot for the new persistent-volume deployment."""
import sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import paramiko

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 31532
PW   = 'MirageVM2026!'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
for i in range(5):
    try:
        c.connect(HOST, port=PORT, username='root', password=PW, timeout=20, banner_timeout=30)
        print('Connected!'); break
    except Exception as e:
        print(f'  attempt {i+1}: {e}'); time.sleep(8)
else:
    print('CANNOT CONNECT'); sys.exit(1)

def run(cmd, timeout=20):
    try:
        _, o, _ = c.exec_command(cmd, timeout=timeout)
        return o.read().decode('utf-8','replace').strip()
    except Exception as ex:
        return f'[err: {ex}]'

print('\n=== UPTIME ===');      print(run('uptime'))
print('\n=== PID1 AGE (s) ==='); print(run('ps -p 1 -o etimes='))
print('\n=== DISK: / and /data ==='); print(run('df -h / /data 2>/dev/null'))
print('\n=== GPU ===');          print(run('nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>/dev/null'))
print('\n=== /data/state ===');  print(run('ls -la /data/state/ 2>/dev/null || echo "(empty)"'))
print('\n=== /data/logs ===');   print(run('ls -lh /data/logs/ 2>/dev/null || echo "(empty)"'))
print('\n=== supervise.log ==='); print(run('cat /data/logs/supervise.log 2>/dev/null || echo "(no file)"'))
print('\n=== pipeline log ==='); print(run('tail -20 $(ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1) 2>/dev/null || echo "(no pipeline log)"'))
print('\n=== watchdog last 3 ==='); print(run('tail -3 /data/logs/watchdog.log 2>/dev/null || echo "(no watchdog)"'))
print('\n=== running procs ==='); print(run('ps aux --no-header | grep -v "ps aux" | grep -E "python|bash /data" || echo "(none)"'))
c.close()
