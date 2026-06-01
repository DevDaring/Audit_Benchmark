"""Research validity audit: prompts, dataset, results, design alignment."""
import json
import re
import sys

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"

AUDIT_SCRIPT = r"""
import json, re, subprocess
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
from Dataset.validate_pentad import assert_production_ready
from Dataset.gold_utils import is_scorable_gold

SEEDS = Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds')
MIRAGE = Path('/data/Audit_Benchmark/Code/mirage')
df = pd.read_parquet(SEEDS / 'pentad_dataset.parquet')
audit = df[df['seed_source'].astype(str).str.lower().isin({'bbq', 'crows_pairs', 'stereoset'})]

issues = []
warnings = []

print('=== 1. PENTAD STRUCTURE ===')
print('rows', len(audit), 'seeds', audit['seed_id'].nunique())
print('sources', audit['seed_source'].value_counts().to_dict())
print('slots', audit['slot'].value_counts().to_dict())

try:
    assert_production_ready(df)
    print('assert_production_ready: PASS')
except Exception as e:
    issues.append(f'assert_production_ready: {e}')
    print('assert_production_ready: FAIL', e)

# Spec seed targets
SPEC = {'bbq': 270, 'crows_pairs': 200, 'stereoset': 200}
actual = {s: int((audit['seed_source'].str.lower() == s).sum() / 12) for s in SPEC}
print('\n=== 2. SEED COUNTS vs SPEC (per-source unique seeds) ===')
for src, target in SPEC.items():
    n = audit[audit['seed_source'].str.lower() == src]['seed_id'].nunique()
    status = 'OK' if n >= {'bbq': 200, 'crows_pairs': 150, 'stereoset': 150}[src] else 'SHORT'
    if n < target:
        warnings.append(f'{src}: {n} seeds vs spec target {target}')
    print(f'  {src}: {n} (target {target}) [{status}]')

exc_path = SEEDS / 'excluded_seeds.json'
if exc_path.exists():
    ex = json.loads(exc_path.read_text())
    n_ex = len(ex) if isinstance(ex, list) else len(ex.get('seeds', ex))
    print(f'  excluded: {n_ex} seeds documented')

# Gold
print('\n=== 3. GOLD ANSWER VALIDITY ===')
slot_a = audit[audit['slot'] == 'a']
for src in ['bbq', 'crows_pairs', 'stereoset']:
    sub = slot_a[slot_a['seed_source'].str.lower() == src]
    sc = sub.apply(lambda r: is_scorable_gold(r['gold_answer'], r['seed_source']), axis=1).sum()
    print(f'  {src} slot-a scorable gold: {sc}/{len(sub)}')
    if sc < len(sub):
        issues.append(f'{src}: {len(sub)-sc} unscorable gold on slot-a')

# Prompt quality
print('\n=== 4. PROMPT QUALITY ===')
sentinel = re.compile(r'^(None|none|nan|NaN|null|NULL|na|NA)$', re.I)
bad = audit[audit['prompt_text'].astype(str).str.strip().str.match(sentinel)]
print(f'  sentinel prompts: {len(bad)}')
if len(bad):
    issues.append(f'{len(bad)} sentinel prompts')

empty = audit[audit['prompt_text'].astype(str).str.strip() == '']
print(f'  empty prompts: {len(empty)}')

# slot b != a
ident_ba = sum(
    1 for _, g in audit.groupby('seed_id')
    if (g[g['slot']=='a']['prompt_text'].iloc[0] if (g['slot']=='a').any() else '') ==
       (g[g['slot']=='b']['prompt_text'].iloc[0] if (g['slot']=='b').any() else '')
)
print(f'  identical slot-a/b seeds: {ident_ba}')
if ident_ba > 0:
    warnings.append(f'{ident_ba} seeds have identical slot-a and slot-b')

# slot c distinct
dup_c = sum(
    1 for _, g in audit.groupby('seed_id')
    if len(set(g[g['slot']=='c']['prompt_text'].astype(str))) < 5
)
print(f'  seeds with non-distinct slot-c: {dup_c}')
if dup_c:
    issues.append(f'{dup_c} seeds with duplicate slot-c variants')

# d/e embed
missing_embed = 0
for sid, g in audit.groupby('seed_id'):
    a_rows = g[g['slot'] == 'a']
    if a_rows.empty:
        continue
    a_text = str(a_rows.iloc[0]['prompt_text']).strip()
    if len(a_text) < 20:
        continue
    snip = a_text[:80]
    for _, r in g[g['slot'].isin(['d', 'e'])].iterrows():
        if snip not in str(r['prompt_text']):
            missing_embed += 1
print(f'  d/e rows missing slot-a embed: {missing_embed}')
if missing_embed:
    issues.append(f'{missing_embed} d/e rows missing slot-a embed')

# BBQ MCQ
bbq = audit[audit['seed_source'].str.lower() == 'bbq']
bbq_ok = bbq['prompt_text'].astype(str).str.contains(r'\(A\)|\(B\)|\(C\)', regex=True, case=False).sum()
print(f'  BBQ rows with MCQ options: {bbq_ok}/{len(bbq)}')

# WinoBias
wino = df[df['seed_source'].astype(str).str.lower() == 'winobias']
print(f'  WinoBias in pentad: {len(wino)} (expect 0)')
if len(wino):
    issues.append(f'{len(wino)} WinoBias rows in main pentad')

# Dev overlap
dev = pd.read_parquet(SEEDS / 'dev_seeds.parquet') if (SEEDS / 'dev_seeds.parquet').exists() else pd.DataFrame()
if len(dev):
    overlap = set(dev['seed_id']) & set(audit['seed_id'])
    print(f'  dev seeds in main pentad: {len(overlap)} (expect 0 for holdout)')
    if overlap:
        warnings.append(f'{len(overlap)} dev seeds leaked into main pentad')

# Samples
print('\n=== 5. SAMPLE PROMPTS ===')
for src in ['bbq', 'crows_pairs', 'stereoset']:
    sub = audit[audit['seed_source'].str.lower() == src]
    if sub.empty:
        continue
    sid = sub['seed_id'].iloc[0]
    g = audit[audit['seed_id'] == sid]
    print(f'\n--- {src} | {sid} ---')
    for slot in ['a', 'b']:
        r = g[g['slot'] == slot].iloc[0]
        print(f'  [{slot}] gold={r["gold_answer"]!r}')
        print(f'       {str(r["prompt_text"])[:200]}')
    cs = g[g['slot'] == 'c']
    print(f'  [c] {len(cs)} variants; unique={cs["prompt_text"].nunique()}')
    for _, r in g[g['slot'] == 'd'].iterrows():
        print(f'  [d/{r["subvariant"]}] {str(r["prompt_text"])[:180]}...')
    for _, r in g[g['slot'] == 'e'].iterrows():
        print(f'  [e/{r["subvariant"]}] {str(r["prompt_text"])[:180]}...')

# Manifest
print('\n=== 6. MANIFEST ===')
mp = SEEDS / 'pentad_manifest.json'
if mp.exists():
    print(mp.read_text()[:400])

# GPU results
print('\n=== 7. GPU RESULTS (if any) ===')
res = MIRAGE / 'results'
for fname in ['behavioral_results.parquet', 'cdva_results.parquet', 'tau_calibration.json']:
    p = res / fname
    if not p.exists():
        print(f'  {fname}: MISSING')
        continue
    if p.suffix == '.parquet':
        rdf = pd.read_parquet(p)
        print(f'  {fname}: {len(rdf)} rows')
        if 'model_name' in rdf.columns:
            print(f'    models: {sorted(rdf.model_name.unique())}')
        if 'success_flag' in rdf.columns:
            sr = rdf['success_flag'].mean()
            print(f'    success_rate: {sr:.3f}')
            if sr < 0.9:
                warnings.append(f'{fname} success_rate only {sr:.1%}')
        if 'failure_reason' in rdf.columns:
            print(f'    failure_reasons: {rdf.failure_reason.value_counts().head(5).to_dict()}')
    else:
        print(f'  {fname}: {p.read_text()[:200]}')

# Progress
try:
    out = subprocess.check_output(['grep', 'prompts done', '/data/logs/pipeline_attempt_1.log'], text=True)
    print('\n=== 8. GPU PROGRESS ===')
    print(out.strip().split('\n')[-1])
except Exception:
    pass

print('\n=== SUMMARY ===')
print('ISSUES:', len(issues))
for i in issues:
    print('  [ISSUE]', i)
print('WARNINGS:', len(warnings))
for w in warnings:
    print('  [WARN]', w)
"""


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)
    cmd = f"/data/venv/bin/python - <<'PY'\n{AUDIT_SCRIPT}\nPY"
    _, o, e = c.exec_command(cmd, timeout=180)
    print((o.read() + e.read()).decode("utf-8", "replace"))
    c.close()


if __name__ == "__main__":
    main()
