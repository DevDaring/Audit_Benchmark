"""Inspect Qwen parse failures on VM."""
from __future__ import annotations

import sys

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"

CMD = r"""
ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1 | xargs tail -15
echo "---"
/data/venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet')
fail = df[(df.model_name == 'qwen2.5-7b-instruct') & (df.sample_index == 0) & (~df.success_flag)]
print('failures', len(fail))
if len(fail):
    r = fail.iloc[0]
    print('slot', r.slot, 'reason', r.failure_reason)
    raw = str(r.raw_response)
    print('raw_len', len(raw))
    print('raw[:600]', raw[:600])
    import sys
    sys.path.insert(0, '/data/Audit_Benchmark/Code/mirage')
    from parse_utils import parse_model_response
    print('new_parser', parse_model_response(raw))
PY
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PW, timeout=45)
    _, o, _ = c.exec_command(CMD, timeout=90)
    print(o.read().decode())
    c.close()


if __name__ == "__main__":
    main()
