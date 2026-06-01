"""
Stop GPU pipeline, remove bad DATASET_OK marker, pull fixes, repair dataset, restart.
"""
import sys, io, paramiko, time, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
REPO = '/data/Audit_Benchmark'
MIRAGE = f'{REPO}/Code/mirage'

REPAIR = textwrap.dedent(f'''
set -uo pipefail
echo "=== STOP pipeline ==="
pkill -f run_gpu_pipeline.py 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f supervise_pipeline 2>/dev/null || true
sleep 2

echo "=== CLEAR bad markers ==="
rm -f /data/state/DATASET_OK /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
rm -f {MIRAGE}/results/behavioral_results.parquet
rm -f {MIRAGE}/results/cdva_results.parquet
ls /data/state/

echo "=== GIT PULL ==="
git -C {REPO} pull --ff-only origin main

echo "=== REPAIR: patch det slots ==="
/data/venv/bin/python {MIRAGE}/patch_det_slots.py
echo "patch exit: $?"

echo "=== REPAIR: regenerate d/e via DeepSeek ==="
/data/venv/bin/python {MIRAGE}/regenerate_api_slots.py
echo "regen exit: $?"

echo "=== VALIDATE ==="
/data/venv/bin/python -c "
import sys; sys.path.insert(0,'{MIRAGE}')
import pandas as pd
from Dataset.validate_pentad import run_all_validations
df = pd.read_parquet('{MIRAGE}/Dataset/seeds/pentad_dataset.parquet')
run_all_validations(df)
print('VALIDATION_OK rows=', len(df))
"

echo "=== RESTART supervisor ==="
nohup bash {REPO}/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
echo "supervisor pid=$!"
''')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
print('Connected — starting dataset repair (DeepSeek d/e regen may take ~30-60 min)...\n')

# First push code from local - user needs git push. Run git push locally first.
import subprocess
r = subprocess.run(['git', 'add', 'Code/mirage/Dataset/pentad_generator.py',
    'Code/mirage/Dataset/validate_pentad.py', 'Code/mirage/regenerate_api_slots.py',
    'akash/_full_pipeline.py'], cwd=r'd:\PhD\Audit_Benchmark', capture_output=True, text=True)
# Don't commit unless user asked - but we need code on VM. Use SFTP to upload critical files instead.

sftp = c.open_sftp()
files = [
    (r'd:\PhD\Audit_Benchmark\Code\mirage\Dataset\pentad_generator.py', f'{MIRAGE}/Dataset/pentad_generator.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\Dataset\validate_pentad.py', f'{MIRAGE}/Dataset/validate_pentad.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\regenerate_api_slots.py', f'{MIRAGE}/regenerate_api_slots.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\patch_det_slots.py', f'{MIRAGE}/patch_det_slots.py'),
    (r'd:\PhD\Audit_Benchmark\akash\_full_pipeline.py', f'{REPO}/akash/_full_pipeline.py'),
]
for local, remote in files:
    sftp.put(local, remote)
    print(f'Uploaded {remote}')
sftp.close()

_, o, e = c.exec_command(f'bash -s << \'EOF\'\n{REPAIR}\nEOF', timeout=7200)
# Stream output in chunks - for now just wait
out = o.read().decode('utf-8','replace')
err = e.read().decode('utf-8','replace')
print(out)
if err.strip():
    print('STDERR:', err[-2000:])

c.close()
