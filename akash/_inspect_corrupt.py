"""Inspect corrupted pentad rows on VM."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'

CODE = textwrap.dedent('''
import pandas as pd
df = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")

# Sentinel prompts
sentinel = df[df["prompt_text"].astype(str).str.strip().isin(["None","none","nan","null"])]
print("Sentinel rows:", len(sentinel))
print("By slot:", sentinel["slot"].value_counts().to_dict())
print("By source:", sentinel["seed_source"].value_counts().to_dict())
if len(sentinel):
    r = sentinel.iloc[0]
    print("Sample:", r["seed_id"], r["slot"], r["subvariant"], repr(r["prompt_text"][:80]))

# StereoSet slot-a sample
st = df[(df["seed_source"]=="stereoset") & (df["slot"]=="a")].head(2)
print("\\nStereoSet slot-a samples:")
for _, r in st.iterrows():
    print(" ", r["seed_id"], ":", repr(str(r["prompt_text"])[:200]))

# BBQ slot-a vs slot-c
bbq_sid = df[(df["seed_source"]=="bbq") & (df["slot"]=="a")].iloc[0]["seed_id"]
g = df[df["seed_id"]==bbq_sid]
print("\\nBBQ seed", bbq_sid)
for _, r in g.iterrows():
    print(f"  {r['slot']}/{r['subvariant']}: {str(r['prompt_text'])[:120]}")

# Check git commit on VM
import subprocess
r = subprocess.run(["git","-C","/data/Audit_Benchmark","log","-1","--oneline"], capture_output=True, text=True)
print("\\nGit:", r.stdout.strip())
''')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/inspect_corrupt.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/inspect_corrupt.py', timeout=60)
print(o.read().decode('utf-8','replace'))
c.close()
