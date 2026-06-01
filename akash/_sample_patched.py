import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
CODE = textwrap.dedent('''
import pandas as pd
df = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
bbq_sid = df[(df["seed_source"]=="bbq")&(df["slot"]=="a")].iloc[0]["seed_id"]
g = df[df["seed_id"]==bbq_sid].sort_values(["slot","subvariant"])
print("=== BBQ", bbq_sid, "===")
for _, r in g.iterrows():
    print(f"\\n[{r['slot']}/{r['subvariant']}] gold={r.get('gold_answer','')}")
    print(str(r["prompt_text"])[:300])
print("\\n=== STEREOSET sample ===")
st_sid = df[(df["seed_source"]=="stereoset")&(df["slot"]=="a")].iloc[0]["seed_id"]
g2 = df[df["seed_id"]==st_sid]
for _, r in g2[g2["slot"].isin(["a","b","c"])].head(7).iterrows():
    print(f"[{r['slot']}/{r['subvariant']}]: {str(r['prompt_text'])[:150]}")
print("c unique:", g2[g2["slot"]=="c"]["prompt_text"].nunique())
''')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/sample_patched.py','w') as f: f.write(CODE)
sftp.close()
_, o, _ = c.exec_command('/data/venv/bin/python /data/sample_patched.py', timeout=30)
print(o.read().decode('utf-8','replace'))
c.close()
