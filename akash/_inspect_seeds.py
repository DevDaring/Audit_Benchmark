"""Inspect seeds.parquet stereoset rows on VM."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import sys, json
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
from Dataset.pentad_generator import _build_full_prompt, _parse_stereoset_sentences

seeds = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet")
print("Seed counts:", seeds["seed_source"].value_counts().to_dict())

st = seeds[seeds["seed_source"]=="stereoset"].head(3)
for _, r in st.iterrows():
    d = r.to_dict()
    pairs = _parse_stereoset_sentences(d.get("sentences"))
    pt, gold = _build_full_prompt(d)
    print("\\n---", d["seed_id"], "---")
    print("context:", repr(str(d.get("context",""))[:80]))
    print("sentences parsed:", len(pairs), pairs[:2] if pairs else "EMPTY")
    print("slot-a prompt:", repr(pt[:150]))
    print("gold:", gold)
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/inspect_seeds.py','w') as f: f.write(CODE)
sftp.close()
_, o, e = c.exec_command('/data/venv/bin/python /data/inspect_seeds.py', timeout=60)
print(o.read().decode('utf-8','replace'))
if e.read().decode('utf-8','replace').strip():
    print('ERR:', e.read()[:500])
c.close()
