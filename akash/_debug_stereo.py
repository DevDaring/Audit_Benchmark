"""Debug StereoSet sentences structure on VM."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import json, sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
seeds = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet")
st = seeds[seeds["seed_source"]=="stereoset"].iloc[0]
raw = st["sentences"]
print("type:", type(raw))
print("repr:", repr(raw)[:500])
if isinstance(raw, str):
    try:
        parsed = json.loads(raw)
        print("json type:", type(parsed))
        print("json sample:", parsed[:2] if isinstance(parsed, list) else parsed)
    except Exception as e:
        print("json err:", e)
elif hasattr(raw, "__iter__"):
    try:
        lst = list(raw)
        print("list len:", len(lst))
        print("first item type:", type(lst[0]) if lst else None)
        print("first item:", lst[0] if lst else None)
    except Exception as e:
        print("iter err:", e)
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/debug_stereo.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/debug_stereo.py', timeout=60)
print(o.read().decode('utf-8','replace'))
c.close()
