"""
1. Stop the running GPU pipeline.
2. Patch pentad_dataset.parquet — add gold_answer column by re-running
   _build_full_prompt() on each seed using seeds.parquet as source of truth.
3. Delete GPU_PIPELINE_OK marker so pipeline re-runs from GPU step.
4. Verify the patch, then restart supervisor.
"""
import sys, io, paramiko, time, textwrap
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST, PORT, PW = 'provider.a100.dsm.val.akash.pub', 31532, 'MirageVM2026!'
REPO = '/data/Audit_Benchmark'

PATCH_CODE = textwrap.dedent("""
import sys, pathlib, logging
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

SEEDS_DIR = pathlib.Path('/data/Audit_Benchmark/Code/mirage/Dataset/seeds')
PENTAD    = SEEDS_DIR / 'pentad_dataset.parquet'
SEEDS_P   = SEEDS_DIR / 'seeds.parquet'

# ------------------------------------------------------------------
# Safety check
# ------------------------------------------------------------------
if not PENTAD.exists():
    print('ERROR: pentad_dataset.parquet not found'); sys.exit(1)
if not SEEDS_P.exists():
    print('ERROR: seeds.parquet not found — cannot rebuild gold_answer'); sys.exit(1)

pentad_df = pd.read_parquet(PENTAD)
seeds_df  = pd.read_parquet(SEEDS_P)

if 'gold_answer' in pentad_df.columns:
    n_empty = (pentad_df['gold_answer'].isna() | (pentad_df['gold_answer'].str.strip()=='')).sum()
    if n_empty == 0:
        print('gold_answer already fully populated — nothing to patch')
        sys.exit(0)
    log.info('gold_answer column exists but %d rows are empty — re-patching', n_empty)

log.info('Pentad rows: %d | seeds: %d', len(pentad_df), len(seeds_df))

# ------------------------------------------------------------------
# Build gold_answer lookup from seeds using pentad_generator logic
# ------------------------------------------------------------------
from Dataset.pentad_generator import _build_full_prompt

seed_to_gold = {}
for _, row in seeds_df.iterrows():
    d = row.to_dict()
    _, gold = _build_full_prompt(d)
    seed_to_gold[d['seed_id']] = gold

missing_seeds = set(pentad_df['seed_id'].unique()) - set(seed_to_gold.keys())
if missing_seeds:
    log.warning('%d seed_ids in pentad not found in seeds.parquet; defaulting to "unknown"', len(missing_seeds))
    for sid in missing_seeds:
        seed_to_gold[sid] = 'unknown'

# Map gold_answer to every pentad row via seed_id
pentad_df['gold_answer'] = pentad_df['seed_id'].map(seed_to_gold).fillna('unknown')

# Validate
null_gold = (pentad_df['gold_answer'].isna() | (pentad_df['gold_answer'].str.strip()=='')).sum()
unknown_gold = (pentad_df['gold_answer'] == 'unknown').sum()
log.info('After patch: %d null, %d "unknown" (WinoBias expected), %d populated',
         null_gold, unknown_gold, len(pentad_df) - null_gold - unknown_gold)

# ------------------------------------------------------------------
# Backup old file and save patched version
# ------------------------------------------------------------------
backup = PENTAD.with_suffix('.parquet.bak')
import shutil; shutil.copy2(PENTAD, backup)
log.info('Backup saved to %s', backup)

pentad_df.to_parquet(PENTAD, index=False)
log.info('Patched pentad_dataset.parquet saved (%d rows, %d columns)', len(pentad_df), len(pentad_df.columns))

# ------------------------------------------------------------------
# Verify
# ------------------------------------------------------------------
check = pd.read_parquet(PENTAD)
assert 'gold_answer' in check.columns, 'FAIL: gold_answer still missing after save'
log.info('Verification PASS: gold_answer column present with %d non-null values',
         check['gold_answer'].notna().sum())
print('PATCH_OK')
""")

STOP_AND_RESTART = textwrap.dedent(f"""
set -uo pipefail
echo "=== Stopping running GPU pipeline ==="
pkill -f "run_gpu_pipeline.py" 2>/dev/null && echo "Killed run_gpu_pipeline" || echo "Not running"
pkill -f "_full_pipeline.py"   2>/dev/null && echo "Killed _full_pipeline"   || echo "Not running"
pkill -f "supervise_pipeline"  2>/dev/null && echo "Killed supervisor"       || echo "Not running"
sleep 3

echo ""
echo "=== Removing GPU_PIPELINE_OK marker (force re-run of GPU step) ==="
rm -f /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
ls /data/state/

echo ""
echo "=== Removing partial behavioral/cdva results (if any) ==="
rm -f /data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet
rm -f /data/Audit_Benchmark/Code/mirage/results/tau_calibration.json

echo ""
echo "=== git pull ==="
git -C {REPO} pull --ff-only origin main 2>&1 || true

echo ""
echo "=== Restarting supervisor ==="
nohup bash {REPO}/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
echo "Supervisor PID=$!"
echo "RESTART_OK"
""")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password=PW, timeout=25)
print(f"Connected to {HOST}:{PORT}\n")

# Step 1: Upload and run patch script
print("=== STEP 1: Patching pentad_dataset.parquet to add gold_answer ===")
sftp = c.open_sftp()
with sftp.open('/data/patch_gold.py', 'w') as f:
    f.write(PATCH_CODE)
sftp.close()

_, out, err = c.exec_command('/data/venv/bin/python /data/patch_gold.py', timeout=120)
patch_out = out.read().decode('utf-8','replace')
patch_err = err.read().decode('utf-8','replace')
print(patch_out)
noisy = {'FutureWarning','DeprecationWarning','UserWarning','_python_version','api_core'}
real_err = [l for l in patch_err.splitlines() if not any(n in l for n in noisy)]
if real_err:
    print("Patch stderr:", '\n'.join(real_err[:15]))

if 'PATCH_OK' not in patch_out and 'already fully populated' not in patch_out:
    print("\nERROR: Patch script failed — NOT restarting pipeline")
    c.close(); sys.exit(1)

# Step 2: Stop pipeline, clean markers, restart
print("\n=== STEP 2: Stop → clean markers → restart supervisor ===")
_, out, err = c.exec_command(f'bash -s << \'HEREDOC\'\n{STOP_AND_RESTART}\nHEREDOC', timeout=60)
restart_out = out.read().decode('utf-8','replace')
restart_err = err.read().decode('utf-8','replace')
print(restart_out)
if restart_err.strip():
    print("Restart stderr:", restart_err[:300])

# Step 3: Quick verify
print("\n=== STEP 3: Verify patch and check supervisor started ===")
time.sleep(20)
_, out, _ = c.exec_command(
    '/data/venv/bin/python -c "'
    'import pandas as pd; df=pd.read_parquet(chr(39)/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquetchr(39));'
    'print(chr(39)gold_answer in columns:chr(39),chr(39)gold_answerchr(39) in df.columns,'
    'chr(39)| non-null:chr(39),df[chr(39)gold_answerchr(39)].notna().sum(),'
    'chr(39)| rows:chr(39),len(df))"',
    timeout=30
)

# use sftp-based verification instead
sftp2 = c.open_sftp()
verify_code = """
import pandas as pd
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
has_col = 'gold_answer' in df.columns
non_null = int(df['gold_answer'].notna().sum()) if has_col else 0
unknown  = int((df['gold_answer'] == 'unknown').sum()) if has_col else 0
print(f'gold_answer present: {has_col} | non_null: {non_null} | unknown(WinoBias): {unknown} | total: {len(df)}')
print('Columns:', list(df.columns))
"""
with sftp2.open('/data/verify_patch.py', 'w') as f:
    f.write(verify_code)
sftp2.close()

_, out, err = c.exec_command('/data/venv/bin/python /data/verify_patch.py', timeout=30)
print(out.read().decode('utf-8','replace'))

print("\n=== STEP 4: Check supervisor started ===")
time.sleep(15)
_, out, _ = c.exec_command('tail -8 /data/logs/supervise.log', timeout=15)
print(out.read().decode('utf-8','replace'))
_, out, _ = c.exec_command('ls /data/state/', timeout=10)
print('State markers:', out.read().decode('utf-8','replace').strip())

c.close()
print("\nMonitor: python akash/_progress_check.py")
