import sys, io, paramiko, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
print("Connected")
for cmd in [
    'ls /data/state/',
    'tail -15 /data/logs/supervise.log',
    'ps aux | grep pipeline | grep -v grep',
    'tail -10 /data/logs/pipeline_attempt_2.log 2>/dev/null || tail -10 /data/logs/pipeline_attempt_1.log 2>/dev/null || echo "no log yet"',
]:
    _, o, _ = c.exec_command(cmd, timeout=15)
    txt = o.read().decode('utf-8','replace')
    print(f'\n--- {cmd} ---\n{txt}')
c.close()
