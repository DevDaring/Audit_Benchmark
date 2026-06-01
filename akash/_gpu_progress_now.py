"""Quick GPU progress check on VM."""
import paramiko
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REMOTE = r"""
echo "=== age ==="
ps -p 1 -o etimes=
echo "=== markers ==="
ls -la /data/state/
echo "=== procs ==="
pgrep -af 'run_gpu_pipeline|supervise_pipeline|_full_pipeline|regenerate_api' || echo none
echo "=== pentad ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
from Dataset.validate_pentad import assert_production_ready
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
assert_production_ready(df)
print('PRODUCTION OK', len(df), 'rows', df.seed_id.nunique(), 'seeds')
PY
echo "=== behavioral ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path('/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet')
if not p.exists():
    print('no behavioral yet')
else:
    df = pd.read_parquet(p)
    print('rows', len(df), 'models', sorted(df.model_name.unique()))
    for m in sorted(df.model_name.unique()):
        sub = df[(df.model_name == m) & (df.sample_index == 0)]
        n_seeds = sub.seed_id.nunique()
        print(f'  {m}: {n_seeds}/596 seeds, {len(sub)} rows')
PY
echo "=== cdva ==="
ls -la /data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet 2>/dev/null || echo missing
echo "=== gpu log milestones ==="
grep -E 'prompts done|Step [0-9]/4|System role|FAILED GPU|model loaded|Behavioral evaluation' /data/logs/pipeline_attempt_*.log 2>/dev/null | tail -20
echo "=== live tail ==="
tail -6 /data/logs/pipeline_attempt_1.log 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!", timeout=45)
_, o, _ = c.exec_command(f"bash -s << 'EOF'\n{REMOTE}\nEOF", timeout=120)
print(o.read().decode())
c.close()
