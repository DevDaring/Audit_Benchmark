import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
seeds = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet")
raw = seeds[seeds["seed_source"]=="stereoset"].iloc[0]["sentences"]
for k,v in raw.items():
    if hasattr(v, "tolist"):
        print(k, ":", v.tolist()[:3] if len(v)>0 else v)
    else:
        print(k, ":", v)
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/debug_stereo3.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/debug_stereo3.py', timeout=60)
print(o.read().decode('utf-8','replace'))
c.close()
