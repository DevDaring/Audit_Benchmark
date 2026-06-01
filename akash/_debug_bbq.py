import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
seeds = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet")
bbq = seeds[seeds["seed_source"]=="bbq"].iloc[0]
print("BBQ cols:", [c for c in bbq.index if bbq[c] is not None][:15])
print("context:", str(bbq.get("context",""))[:100])
print("question:", bbq.get("question"))
print("ans0/1/2:", bbq.get("ans0"), bbq.get("ans1"), bbq.get("ans2"))
print("label:", bbq.get("label"))
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/debug_bbq.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/debug_bbq.py', timeout=30)
print(o.read().decode('utf-8','replace'))
c.close()
