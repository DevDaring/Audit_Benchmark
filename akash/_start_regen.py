"""Start DeepSeek d/e regeneration in tmux on VM."""
import sys, io, paramiko, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
MIRAGE = '/data/Audit_Benchmark/Code/mirage'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)

# Upload regenerate script
sftp = c.open_sftp()
sftp.put(r'd:\PhD\Audit_Benchmark\Code\mirage\regenerate_api_slots.py',
         f'{MIRAGE}/regenerate_api_slots.py')
sftp.close()

cmd = f'''
pkill -f run_gpu_pipeline 2>/dev/null; pkill -f _full_pipeline 2>/dev/null; pkill -f supervise_pipeline 2>/dev/null
tmux kill-session -t regen 2>/dev/null || true
tmux new-session -d -s regen "/data/venv/bin/python {MIRAGE}/regenerate_api_slots.py 2>&1 | tee /data/logs/regenerate_api_slots.log; echo EXIT=$? >> /data/logs/regenerate_api_slots.log"
echo "tmux session regen started"
sleep 2
tail -5 /data/logs/regenerate_api_slots.log 2>/dev/null || echo waiting...
'''
_, o, _ = c.exec_command(cmd, timeout=30)
print(o.read().decode('utf-8','replace'))
c.close()
print('Monitor: tail -f /data/logs/regenerate_api_slots.log')
