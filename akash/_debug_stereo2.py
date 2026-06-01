"""Full StereoSet sentences dict structure."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import json, sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
seeds = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet")
raw = seeds[seeds["seed_source"]=="stereoset"].iloc[0]["sentences"]
print("keys:", list(raw.keys()) if isinstance(raw, dict) else "not dict")
if isinstance(raw, dict) and "labels" in raw:
    labels = raw["labels"]
    print("labels type:", type(labels), "len:", len(labels))
    first = labels[0]
    print("first label type:", type(first))
    if isinstance(first, dict):
        print("first label keys:", list(first.keys()))
        for k,v in first.items():
            print(" ", k, ":", type(v), repr(v)[:120] if not hasattr(v,'shape') else f"array shape {getattr(v,'shape',None)}")
        if "sentence" in first:
            print("sentence:", first["sentence"])
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/debug_stereo2.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/debug_stereo2.py', timeout=60)
print(o.read().decode('utf-8','replace'))
c.close()
