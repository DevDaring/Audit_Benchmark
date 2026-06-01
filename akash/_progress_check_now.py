"""Live VM progress check with ETA."""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"

REMOTE = r"""
echo '=== MARKERS ==='
ls -la /data/state/ 2>/dev/null
echo '=== PROCS ==='
pgrep -af 'supervise_pipeline|run_gpu_pipeline|_full_pipeline' || echo none
echo '=== GPU ==='
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo n/a
echo '=== LOG TAIL ==='
L=$(ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1)
echo "log=$L"
tail -15 "$L" 2>/dev/null
echo '=== GEMMA LINES ==='
grep 'gemma.*prompts done' "$L" 2>/dev/null | tail -6
echo '=== BEHAVIORAL ==='
/data/venv/bin/python - <<'PY'
import pandas as pd, sys
sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
from results_utils import dedup_behavioral, reparse_failed_rows
from Dataset.validate_pentad import assert_production_ready
p = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
assert_production_ready(p)
print('pentad OK', p.seed_id.nunique(), 'seeds')
df = dedup_behavioral(reparse_failed_rows(pd.read_parquet('/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet')))
print('behavioral rows', len(df), 'dup', int(df.duplicated(['prompt_id','model_name','sample_index']).sum()), 'fail', int((~df.success_flag).sum()))
for m in sorted(df.model_name.unique()):
    si0 = df[(df.model_name==m)&(df.sample_index==0)]
    print(m, 'si0', int(si0.success_flag.sum()), '/', len(si0))
import pathlib
print('cdva', pathlib.Path('/data/Audit_Benchmark/Code/mirage/results/cdva_results.parquet').exists())
PY
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PW, timeout=50)
    _, o, _ = c.exec_command(REMOTE, timeout=120)
    text = o.read().decode()
    print(text)

    # Parse gemma progress for ETA
    lines = [ln for ln in text.splitlines() if "gemma" in ln.lower() and "prompts done" in ln]
    if len(lines) >= 2:
        pat = re.compile(
            r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z.*?(\d+)/7152 prompts done.*?sample_index=(\d+)"
        )
        parsed = []
        for ln in lines:
            m = pat.search(ln)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                parsed.append((ts, int(m.group(2)), int(m.group(3))))
        if len(parsed) >= 2 and parsed[-1][2] == 0:
            t0, n0, _ = parsed[-2]
            t1, n1, _ = parsed[-1]
            dt = (t1 - t0).total_seconds()
            dn = max(n1 - n0, 1)
            rate = dn / dt if dt > 0 else 0
            remain = 7152 - n1
            if rate > 0:
                gemma_det_min = remain / rate / 60
                print(f"\n=== ETA HINT ===")
                print(f"Gemma det: {n1}/7152 @ {rate*60:.0f} prompts/min -> ~{gemma_det_min:.0f} min left for det")
    c.close()


if __name__ == "__main__":
    main()
