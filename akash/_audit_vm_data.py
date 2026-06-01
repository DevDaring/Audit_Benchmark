"""Full VM data + code integrity audit."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"
REPO = Path(__file__).resolve().parents[1] / "Code/mirage"
FILES = [
    "parse_utils.py",
    "results_utils.py",
    "GPU_CPU/osm_behavioral.py",
    "GPU_CPU/cdva_patching.py",
    "CPU_Only/scoring.py",
    "CPU_Only/leaderboard.py",
    "CPU_Only/api_behavioral.py",
    "Dataset/category_utils.py",
]

AUDIT = r"""
/data/venv/bin/python - <<'PY'
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
from results_utils import dedup_behavioral, reparse_failed_rows
from Dataset.validate_pentad import assert_production_ready

print('=== PENTAD ===')
p = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
assert_production_ready(p)
print('PRODUCTION_OK', len(p), 'rows', p.seed_id.nunique(), 'seeds')

print('=== BEHAVIORAL on disk ===')
bpath = Path('/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet')
raw = pd.read_parquet(bpath)
print('rows', len(raw))
print('duplicate_triples', int(raw.duplicated(['prompt_id','model_name','sample_index']).sum()))
print('failed_rows', int((~raw.success_flag).sum()))
clean = dedup_behavioral(reparse_failed_rows(raw))
print('after_dedup_reparse_rows', len(clean))
print('after_dedup_failed', int((~clean.success_flag).sum()))
for m in sorted(clean.model_name.unique()):
    for si in sorted(clean.sample_index.unique()):
        sub = clean[(clean.model_name==m) & (clean.sample_index==si)]
        ok = int(sub.success_flag.sum())
        print(f'  {m} si={si}: rows={len(sub)} ok={ok} fail={len(sub)-ok}')
needs_rewrite = len(raw) != len(clean) or raw.duplicated(['prompt_id','model_name','sample_index']).any() or (~reparse_failed_rows(raw).success_flag).sum() != (~clean.success_flag).sum()
print('needs_rewrite', needs_rewrite)
print('=== CDVA ===')
cp = Path('/data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet')
print('exists', cp.exists())
print('=== GPU ===')
import subprocess
print(subprocess.check_output('pgrep -af run_gpu_pipeline || echo stopped', shell=True, text=True).strip())
PY
ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1 | xargs tail -8
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PW, timeout=50)

    print("=== CODE SYNC ===")
    for f in FILES:
        local = REPO / f
        lh = hashlib.md5(local.read_bytes()).hexdigest()
        _, o, _ = c.exec_command(
            f"md5sum /data/Audit_Benchmark/Code/mirage/{f} 2>/dev/null || echo MISSING",
            timeout=30,
        )
        line = o.read().decode().strip()
        rh = line.split()[0] if line and "MISSING" not in line else "MISSING"
        match = "OK" if lh == rh else "MISMATCH"
        print(f"  {match} {f}")

    print("\n=== DATA AUDIT ===")
    _, o, _ = c.exec_command(AUDIT, timeout=120)
    print(o.read().decode())
    c.close()


if __name__ == "__main__":
    main()
