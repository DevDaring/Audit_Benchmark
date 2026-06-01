import sys, io, paramiko
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('provider.a100.dsm.val.akash.pub', 31532, username='root', password='MirageVM2026!', timeout=25)
_, o, _ = c.exec_command('''/data/venv/bin/python -c "
import pandas as pd
df=pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
st=df[(df['seed_source']=='stereoset')&(df['slot']=='a')].iloc[0]
print(st['prompt_text'])
print('---C variants---')
for _,r in df[(df['seed_id']==st['seed_id'])&(df['slot']=='c')].iterrows():
    print(r['subvariant'], ':', r['prompt_text'][-200:])
"''', timeout=30)
print(o.read().decode('utf-8','replace'))
c.close()
