"""Full semantic audit of pentad_dataset.parquet on VM."""
import sys, io, paramiko, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'

AUDIT = textwrap.dedent(r'''
import sys, re, pathlib
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
import pandas as pd
import numpy as np

PENTAD = pathlib.Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
SEEDS  = pathlib.Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/seeds.parquet')
df = pd.read_parquet(PENTAD)
seeds = pd.read_parquet(SEEDS) if SEEDS.exists() else None

issues = []
warns = []

print('='*65)
print('FINAL RESEARCH AUDIT — pentad_dataset.parquet')
print('='*65)
print(f'Rows: {len(df)} | Seeds: {df["seed_id"].nunique()} | Cols: {len(df.columns)}')

# 1 gold_answer
if 'gold_answer' not in df.columns:
    issues.append('CRITICAL: gold_answer column MISSING')
else:
    null = df['gold_answer'].isna().sum()
    empty = (df['gold_answer'].astype(str).str.strip() == '').sum()
    unknown = (df['gold_answer'] == 'unknown').sum()
    print(f'\n[1] gold_answer: null={null} empty={empty} unknown={unknown} populated={len(df)-null-empty}')
    if null or empty:
        issues.append(f'gold_answer has {null+empty} null/empty rows')

# 2 Run validate_pentad
print('\n[2] validate_pentad.run_all_validations')
try:
    from Dataset.validate_pentad import run_all_validations
    run_all_validations(df)
    print('  PASS — all structural + semantic checks')
except Exception as e:
    issues.append(f'validate_pentad FAILED: {e}')
    print(f'  FAIL: {e}')

# 3 slot-b differs from slot-a rate
identical_ab = 0
total = 0
for sid, g in df.groupby('seed_id'):
    a = g[g['subvariant']=='surface']
    b = g[g['subvariant']=='iso_control']
    if a.empty or b.empty: continue
    total += 1
    if str(a.iloc[0]['prompt_text']).strip() == str(b.iloc[0]['prompt_text']).strip():
        identical_ab += 1
rate_ab = identical_ab/total if total else 0
print(f'\n[3] slot-b == slot-a: {identical_ab}/{total} ({rate_ab:.1%})')
if rate_ab > 0.5:
    issues.append(f'slot-b identical to slot-a for {rate_ab:.1%} of seeds')

# 4 slot-c distinctness
degen_c = 0
for sid, g in df.groupby('seed_id'):
    c = g[g['slot']=='c']
    if len(c) >= 2 and c['prompt_text'].nunique() == 1:
        degen_c += 1
rate_c = degen_c/total if total else 0
print(f'[4] slot-c all-identical: {degen_c}/{total} ({rate_c:.1%})')
if rate_c > 0.25:
    issues.append(f'slot-c degenerate for {rate_c:.1%} of seeds')

# 5 DeepSeek d/e embed original slot-a text
missing_embed_d = missing_embed_e = 0
checked_d = checked_e = 0
for sid, g in df.groupby('seed_id'):
    a_rows = g[g['slot']=='a']
    if a_rows.empty: continue
    a_text = str(a_rows.iloc[0]['prompt_text']).strip()
    if len(a_text) < 20: continue
    # use last 80 chars of slot-a as fingerprint (question+options tail)
    fingerprint = a_text[-80:].lower()
    for slot, counter in [('d', 'missing_embed_d'), ('e', 'missing_embed_e')]:
        for _, row in g[g['slot']==slot].iterrows():
            if slot == 'd': checked_d += 1
            else: checked_e += 1
            pt = str(row['prompt_text']).lower()
            if fingerprint not in pt and a_text.lower() not in pt:
                if slot == 'd': missing_embed_d += 1
                else: missing_embed_e += 1

print(f'\n[5] DeepSeek embeds original slot-a text')
print(f'  slot-d: {checked_d - missing_embed_d}/{checked_d} contain slot-a tail ({missing_embed_d} missing)')
print(f'  slot-e: {checked_e - missing_embed_e}/{checked_e} contain slot-a tail ({missing_embed_e} missing)')
if missing_embed_d > checked_d * 0.05:
    warns.append(f'{missing_embed_d}/{checked_d} slot-d prompts may not embed original slot-a')
if missing_embed_e > checked_e * 0.05:
    warns.append(f'{missing_embed_e}/{checked_e} slot-e prompts may not embed original slot-a')

# 6 gold_answer by source
print('\n[6] gold_answer by source (slot-a rows)')
if 'gold_answer' in df.columns:
    a_df = df[df['slot']=='a'][['seed_source','gold_answer']].drop_duplicates('seed_source', keep='first')
    for src in df['seed_source'].unique():
        sub = df[(df['seed_source']==src) & (df['slot']=='a')]
        n = len(sub)
        unk = (sub['gold_answer']=='unknown').sum()
        print(f'  {src:15s} seeds={n:3d} unknown_gold={unk:3d} ({unk/n*100:.0f}%)')

# 7 seed count vs research design
print('\n[7] Seed counts vs design (270 BBQ + 200 CrowS + 200 StereoSet + 200 WinoBias = 870)')
if seeds is not None:
    print(seeds['seed_source'].value_counts().to_string())
    n_main = len(seeds)
    if n_main != 668:
        warns.append(f'seeds.parquet has {n_main} seeds (668 main audit seeds expected)')

# 8 DeepSeek attribution
print('\n[8] DeepSeek generator attribution')
for slot in ['d','e']:
    s = df[df['slot']==slot]
    print(f'  slot {slot}: generated_by={s["generated_by"].value_counts().to_dict()}')
    print(f'         generator_model={s["generator_model"].value_counts().to_dict()}')

# 9 CDVA bias_answer coverage (simulate)
if 'gold_answer' in df.columns:
    c_df = df[df['slot']=='c']
    no_bias = 0
    for sid, g in c_df.groupby('seed_id'):
        gold_vals = g['gold_answer'].dropna().unique()
        if len(gold_vals)==0 or str(gold_vals[0]).strip().lower() in ('','unknown'):
            no_bias += 1
    print(f'\n[9] CDVA bias_answer unavailable (unknown gold): {no_bias}/{df["seed_id"].nunique()} seeds')
    if no_bias > 200:
        warns.append(f'{no_bias} seeds will skip CDVA patching (no bias_answer token)')

# 10 Sample d/e prompts
print('\n[10] Sample DeepSeek prompts (first BBQ seed)')
bbq = df[(df['seed_source']=='bbq') & (df['slot']=='a')].iloc[0]
sid = bbq['seed_id']
a = str(bbq['prompt_text'])[:100]
for slot in ['d','e']:
    row = df[(df['seed_id']==sid) & (df['slot']==slot)].iloc[0]
    print(f'  {slot}/{row["subvariant"]}: {str(row["prompt_text"])[:150]}...')

print('\n' + '='*65)
print('ISSUES (must fix):', len(issues))
for i in issues: print('  [!]', i)
print('WARNINGS (review):', len(warns))
for w in warns: print('  [?]', w)
print('OVERALL:', 'PASS' if not issues else 'FAIL')
print('='*65)
''')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
sftp = c.open_sftp()
with sftp.open('/data/final_audit.py','w') as f: f.write(AUDIT)
sftp.close()
_, o, e = c.exec_command('/data/venv/bin/python /data/final_audit.py', timeout=120)
print(o.read().decode('utf-8','replace'))
err = e.read().decode('utf-8','replace')
if err.strip():
    lines = [l for l in err.splitlines() if 'INFO' in l or 'ERROR' in l or 'WARNING' in l]
    if lines: print('LOG:', '\n'.join(lines[-20:]))
c.close()
