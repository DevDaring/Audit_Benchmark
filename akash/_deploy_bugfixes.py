"""Upload bugfix bundle and restart GPU pipeline from checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"
REPO = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/data/Audit_Benchmark/Code/mirage"

UPLOADS = [
    ("parse_utils.py", f"{REMOTE_ROOT}/parse_utils.py"),
    ("results_utils.py", f"{REMOTE_ROOT}/results_utils.py"),
    ("Dataset/category_utils.py", f"{REMOTE_ROOT}/Dataset/category_utils.py"),
    ("GPU_CPU/osm_behavioral.py", f"{REMOTE_ROOT}/GPU_CPU/osm_behavioral.py"),
    ("GPU_CPU/cdva_patching.py", f"{REMOTE_ROOT}/GPU_CPU/cdva_patching.py"),
    ("CPU_Only/scoring.py", f"{REMOTE_ROOT}/CPU_Only/scoring.py"),
    ("CPU_Only/leaderboard.py", f"{REMOTE_ROOT}/CPU_Only/leaderboard.py"),
    ("CPU_Only/api_behavioral.py", f"{REMOTE_ROOT}/CPU_Only/api_behavioral.py"),
]

REMOTE_CMD = r"""
set -uo pipefail
echo "=== dedup + reparse failed rows ==="
/data/venv/bin/python - <<'PY'
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
from results_utils import dedup_behavioral
from parse_utils import parse_model_response

p = Path("/data/Audit_Benchmark/Code/mirage/results/behavioral_results.parquet")
if p.exists():
    df = pd.read_parquet(p)
    print("before", len(df))
    fixed = 0
    for idx, row in df[~df.success_flag].iterrows():
        ok, ans, conf, rat, method, reason = parse_model_response(str(row.get("raw_response", "")))
        if ok:
            df.at[idx, "success_flag"] = True
            df.at[idx, "parsed_answer"] = ans
            df.at[idx, "parsed_confidence"] = conf
            df.at[idx, "parsed_rationale"] = rat
            df.at[idx, "parse_method"] = method
            df.at[idx, "failure_reason"] = ""
            fixed += 1
    print("reparse_fixed", fixed)
    clean = dedup_behavioral(df)
    clean.to_parquet(p, index=False)
    print("after", len(clean))
    for m in sorted(clean.model_name.unique()):
        sub = clean[(clean.model_name == m) & (clean.sample_index == 0)]
        ok = sub[sub.success_flag == True]
        print(m, "si0 ok", len(ok), "fail", len(sub) - len(ok))
else:
    print("no behavioral yet")
PY
echo "=== stop gpu only (keep markers + checkpoint) ==="
pkill -f run_gpu_pipeline.py 2>/dev/null || true
sleep 3
pgrep -af run_gpu_pipeline || echo gpu stopped — supervisor will restart
"""


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PW, timeout=45)
    sftp = c.open_sftp()
    for rel, remote in UPLOADS:
        local = REPO / "Code/mirage" / rel
        sftp.put(str(local), remote)
        print(f"Uploaded {rel}")
    sftp.close()
    _, o, e = c.exec_command(f"bash -s << 'EOF'\n{REMOTE_CMD}\nEOF", timeout=120)
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print(err)
    c.close()


if __name__ == "__main__":
    main()
