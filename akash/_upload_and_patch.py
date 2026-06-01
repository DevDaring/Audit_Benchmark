"""Upload fixes and run patch_det_slots on VM (fast step)."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
REPO, MIRAGE = '/data/Audit_Benchmark', '/data/Audit_Benchmark/Code/mirage'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)

sftp = c.open_sftp()
for local, remote in [
    (r'd:\PhD\Audit_Benchmark\Code\mirage\Dataset\pentad_generator.py', f'{MIRAGE}/Dataset/pentad_generator.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\Dataset\validate_pentad.py', f'{MIRAGE}/Dataset/validate_pentad.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\regenerate_api_slots.py', f'{MIRAGE}/regenerate_api_slots.py'),
    (r'd:\PhD\Audit_Benchmark\Code\mirage\patch_det_slots.py', f'{MIRAGE}/patch_det_slots.py'),
    (r'd:\PhD\Audit_Benchmark\akash\_full_pipeline.py', f'{REPO}/akash/_full_pipeline.py'),
]:
    sftp.put(local, remote)
sftp.close()
print('Files uploaded')

# Stop pipeline + patch
CMD = textwrap.dedent(f'''
pkill -f run_gpu_pipeline.py 2>/dev/null; pkill -f _full_pipeline.py 2>/dev/null; pkill -f supervise_pipeline 2>/dev/null
rm -f /data/state/DATASET_OK /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
/data/venv/bin/python {MIRAGE}/patch_det_slots.py 2>&1
''')
_, o, e = c.exec_command(CMD, timeout=300)
print(o.read().decode('utf-8','replace'))
print(e.read().decode('utf-8','replace')[-1000:])

# Quick check
CHECK = textwrap.dedent(f'''
import sys; sys.path.insert(0,"{MIRAGE}")
import pandas as pd
df = pd.read_parquet("{MIRAGE}/Dataset/seeds/pentad_dataset.parquet")
sentinel = df[df["prompt_text"].astype(str).str.strip().isin(["None","none","null"])]
print("Sentinel after patch:", len(sentinel))
st = df[(df["seed_source"]=="stereoset")&(df["slot"]=="a")].iloc[0]
print("StereoSet slot-a sample:", str(st["prompt_text"])[:200])
bbq_sid = df[(df["seed_source"]=="bbq")&(df["slot"]=="a")].iloc[0]["seed_id"]
g = df[df["seed_id"]==bbq_sid]
print("BBQ c unique texts:", g[g["slot"]=="c"]["prompt_text"].nunique())
print("BBQ a==b:", g[g["subvariant"]=="surface"].iloc[0]["prompt_text"]==g[g["subvariant"]=="iso_control"].iloc[0]["prompt_text"])
''')
sftp = c.open_sftp()
with sftp.open('/data/check_patch.py','w') as f: f.write(CHECK)
sftp.close()
_, o, _ = c.exec_command(f'/data/venv/bin/python /data/check_patch.py', timeout=60)
print('\n--- POST-PATCH CHECK ---')
print(o.read().decode('utf-8','replace'))
c.close()
