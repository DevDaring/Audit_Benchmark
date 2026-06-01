"""Supplementary slot-b and research design checks."""
import paramiko

SCRIPT = r"""
import re, json
from pathlib import Path
from collections import Counter
import pandas as pd

df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
audit = df[df.seed_source.str.lower().isin(['bbq','crows_pairs','stereoset'])]

def flag_b(text):
    t = str(text).lower()
    issues = []
    if 'person and person' in t:
        issues.append('person_and_person')
    if re.search(r'\bperson man\b', t):
        issues.append('person_man')
    if re.search(r'\bperson woman\b', t):
        issues.append('person_woman')
    return issues

by_src = {}
for sid, g in audit.groupby('seed_id'):
    b = g[g.slot == 'b'].iloc[0]
    fl = flag_b(b['prompt_text'])
    if fl:
        c = by_src.setdefault(b['seed_source'], Counter())
        for f in fl:
            c[f] += 1

print('=== SLOT-B GRAMMAR FLAGS ===')
for src, ctr in sorted(by_src.items()):
    print(src, dict(ctr), 'seeds_flagged', sum(ctr.values()))

print('\n=== GOLD CONSISTENCY ===')
mismatch = sum(
    1 for _, g in audit.groupby('seed_id')
    if g['gold_answer'].astype(str).str.strip().nunique() > 1
)
print('seeds with inconsistent gold:', mismatch)

print('\n=== CATEGORY COVERAGE (slot-a unique seeds) ===')
print(
    audit[audit.slot == 'a']
    .groupby(['seed_source', 'seed_category'])['seed_id']
    .nunique()
    .to_string()
)

wino = Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/winobias_seeds.parquet')
if wino.exists():
    print('\n=== WINOBIAS HELD OUT ===', len(pd.read_parquet(wino)), 'seeds')

ex = json.loads(
    Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/excluded_seeds.json').read_text()
)
print('\n=== EXCLUDED ===', ex.get('n_excluded', '?'), 'seeds documented')

# Progress
import subprocess
try:
    line = subprocess.check_output(
        ['grep', 'prompts done', '/data/logs/pipeline_attempt_1.log'], text=True
    ).strip().split('\n')[-1]
    print('\n=== GPU ===', line)
except Exception:
    pass
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("provider.a100.dsm.val.akash.pub", 31532, username="root", password="MirageVM2026!", timeout=30)
_, o, e = c.exec_command(f"/data/venv/bin/python - <<'PY'\n{SCRIPT}\nPY", timeout=90)
print((o.read() + e.read()).decode("utf-8", "replace"))
c.close()
