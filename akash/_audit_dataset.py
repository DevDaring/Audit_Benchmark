"""
Audit pentad_dataset.parquet against the MIRAGE research design spec.
Uploads audit script to VM and runs it there.
"""
import sys, io, paramiko, time, pathlib, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'

AUDIT_CODE = textwrap.dedent("""
import sys, pathlib
import pandas as pd

PENTAD = pathlib.Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
if not PENTAD.exists():
    print('ERROR: pentad_dataset.parquet not found'); sys.exit(1)

df = pd.read_parquet(PENTAD)
ok = True
SEP = '='*62

print('\\n' + SEP)
print('PENTAD DATASET AUDIT vs MIRAGE research design')
print(SEP)
print(f'Total rows : {len(df)}')
print(f'File size  : {PENTAD.stat().st_size/1024:.1f} KB')
print(f'Columns    : {list(df.columns)}')

# 1. Row counts
expected_full = 668 * 12   # 8016
expected_det  = 668 * 7    # 4676 (a=1,b=1,c=5)
print('\\n[1] ROW COUNT (668 seeds x 12 = 8016 full / x7 = 4676 det-only)')
if len(df) == expected_full:
    print('  PASS  — full 12-slot dataset including DeepSeek slots d and e')
elif len(df) == expected_det:
    print('  WARN  — only deterministic slots (a/b/c) present; d and e MISSING')
    ok = False
else:
    n_seeds = df['seed_id'].nunique()
    print(f'  INFO  — {len(df)} rows / {n_seeds} seeds (may differ from 668 if src has fewer items)')

# 2. Slot distribution
print('\\n[2] SLOT DISTRIBUTION')
sc = df['slot'].value_counts().to_dict()
print(f'  raw counts: {sc}')
n_seeds = df['seed_id'].nunique()
for slot, per_seed in [('a',1),('b',1),('c',5),('d',2),('e',3)]:
    n = sc.get(slot, 0)
    exp = n_seeds * per_seed
    tag = 'PASS' if n == exp else f'FAIL (got {n}, expected {exp})'
    print(f'  slot {slot} : {n:5d}  {tag}')
    if n != exp: ok = False

# 3. DeepSeek slots detail
print('\\n[3] DEEPSEEK SLOTS d AND e')
for slot in ['d','e']:
    sdf = df[df['slot'] == slot]
    if len(sdf) == 0:
        print(f'  slot {slot}: MISSING — DeepSeek prompts NOT in dataset')
        ok = False; continue
    empty = (sdf['prompt_text'].isna() | (sdf['prompt_text'].str.strip() == '')).sum()
    gen_col = sdf['generator_model'].value_counts().to_dict() if 'generator_model' in sdf.columns else 'col missing'
    sub = sdf['subvariant'].value_counts().to_dict()
    print(f'  slot {slot}: {len(sdf)} rows | empty_prompt={empty} | subvariants={sub}')
    print(f'         generator_model: {gen_col}')
    if empty > 0:
        print(f'         WARN: {empty} rows have empty prompt_text'); ok = False

# 4. Duplicate prompt_ids
print('\\n[4] DUPLICATE PROMPT_IDS')
dupes = df['prompt_id'].duplicated().sum()
print(f'  duplicates: {dupes}  -> {"PASS" if dupes==0 else "FAIL"}')
if dupes > 0: ok = False

# 5. Null prompt_text
print('\\n[5] NULL / EMPTY PROMPT_TEXT')
nulls = df['prompt_text'].isna().sum() + (df['prompt_text'].str.strip() == '').sum()
print(f'  null or empty: {nulls}  -> {"PASS" if nulls==0 else "FAIL"}')
if nulls > 0: ok = False

# 6. Required columns
print('\\n[6] REQUIRED COLUMNS')
required = ['seed_id','seed_source','seed_category','prompt_id','slot',
            'subvariant','prompt_text','gold_answer','generated_by','generator_model']
missing = [c for c in required if c not in df.columns]
print(f'  missing: {missing}  -> {"PASS" if not missing else "FAIL"}')
if missing: ok = False

# 7. gold_answer completeness (WinoBias legitimately says 'unknown')
print('\\n[7] GOLD_ANSWER (slots a/b/c only)')
det = df[df['slot'].isin(['a','b','c'])]
null_gold = (det['gold_answer'].isna() | (det['gold_answer'].str.strip()=='') | (det['gold_answer']=='unknown')).sum()
pct = null_gold / len(det) * 100
print(f'  unknown/null: {null_gold}/{len(det)} ({pct:.1f}%)')
print(f'  -> {"PASS" if pct < 30 else "WARN: >30% missing gold"}')

# 8. Slot c: exactly 5 per seed
print('\\n[8] SLOT C: 5 VARIANTS PER SEED')
c_per = df[df['slot']=='c'].groupby('seed_id').size()
perfect = (c_per==5).sum(); bad = (c_per!=5).sum()
print(f'  seeds with exactly 5: {perfect} / {n_seeds}  -> {"PASS" if bad==0 else f"WARN: {bad} seeds incomplete"}')
if bad > 0: ok = False

# 9. Source distribution
print('\\n[9] SOURCE DISTRIBUTION')
print(df.drop_duplicates('seed_id')['seed_source'].value_counts().to_string())

# 10. Sample DeepSeek prompts
print('\\n[10] SAMPLE DEEPSEEK PROMPTS (first row of each subvariant)')
for slot, subv in [('d','d_valid'),('d','d_harmful'),('e','e1'),('e','e2'),('e','e3')]:
    m = df[(df['slot']==slot) & df['subvariant'].str.startswith(subv, na=False)]
    if len(m):
        row = m.iloc[0]
        txt = str(row['prompt_text'])[:130].replace('\\n',' ')
        print(f'  {slot}/{row["subvariant"]}: {txt}...')
    else:
        print(f'  {slot}/{subv}: NOT FOUND')

print()
print(SEP)
print('OVERALL:', 'ALL CHECKS PASS' if ok else 'ISSUES FOUND — see details above')
print(SEP)
""")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)

# Upload audit script as a file then run it
sftp = c.open_sftp()
with sftp.open('/data/audit_dataset.py', 'w') as f:
    f.write(AUDIT_CODE)
sftp.close()

_, stdout, stderr = c.exec_command('/data/venv/bin/python /data/audit_dataset.py', timeout=90)
out = stdout.read().decode('utf-8','replace')
err = stderr.read().decode('utf-8','replace')
print(out)
noisy = {'FutureWarning','DeprecationWarning','UserWarning','_python_version'}
real_err = [l for l in err.splitlines() if not any(n in l for n in noisy)]
if real_err:
    print('STDERR:', '\n'.join(real_err[:15]))
c.close()
