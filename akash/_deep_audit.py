"""Deep audit: slot-b grammar, checkpoint d prompts, code smoke tests on VM."""
import sys
import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"

SCRIPT = r"""
set -euo pipefail
PY=/data/venv/bin/python
MIRAGE=/data/Audit_Benchmark/Code/mirage
cd $MIRAGE

echo "=== PROCESSES ==="
pgrep -af 'regenerate_api|supervise|autonomous_guard|run_gpu' || echo none

echo "=== MARKERS ==="
ls -la /data/state/

echo "=== DET PENTAD VALIDATION ==="
$PY - <<'PY'
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
from Dataset.validate_pentad import (
    run_all_validations, validate_slot_b_grammar, assert_production_ready
)
from Dataset.gold_utils import is_scorable_gold

df = pd.read_parquet("Dataset/seeds/pentad_dataset.parquet")
audit = df[df.seed_source.str.lower().isin(["bbq","crows_pairs","stereoset"])]
print("rows", len(df), "de", int(df.slot.isin(["d","e"]).sum()))
try:
    run_all_validations(df, require_api_slots=False)
    print("det_validation: PASS")
except Exception as e:
    print("det_validation: FAIL", e)
    sys.exit(1)
validate_slot_b_grammar(df)
print("slot_b_grammar: PASS")
try:
    assert_production_ready(df)
    print("production: PASS")
except Exception as e:
    print("production: PENDING", str(e)[:120])
PY

echo "=== CHECKPOINT D PROMPT SAMPLE ==="
$PY - <<'PY'
import json, random
from pathlib import Path
cp = Path("Dataset/seeds/context_shift_checkpoint.json")
if not cp.exists():
    print("no checkpoint")
else:
    ck = json.load(open(cp))
    print("ctx_seeds", len(ck))
    keys = list(ck.keys())
    for sid in random.sample(keys, min(3, len(keys))):
        rows = ck[sid]
        for r in rows:
            t = r["prompt_text"]
            print(f"\n--- {sid} {r['subvariant']} ---")
            print(t[:220].replace("\n", " "))
            # embed check: slot-a snippet should appear
            print("len", len(t))
PY

echo "=== CODE SMOKE ==="
$PY - <<'PY'
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
from Dataset.pentad_generator import _replace_with_neutral_person, _polish_slot_b_text
from Dataset.context_shift_drafter import _PARALLEL_WORKERS
import regenerate_api_slots
assert _replace_with_neutral_person("The Muslim man is very BLANK", "Muslim") == "The person is very BLANK"
assert "People are" in _replace_with_neutral_person("Context: Gentlemen are BLANK", "Gentlemen")
assert hasattr(regenerate_api_slots, "_save_partial_pentad")
print("parallel_workers", _PARALLEL_WORKERS)
print("code_smoke: PASS")
PY

echo "=== METRICS ==="
ls -la results/ 2>/dev/null || echo no results dir
"""

def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(8):
        try:
            c.connect(HOST, PORT, username=USER, password=PW, timeout=25, banner_timeout=45)
            break
        except Exception as e:
            print(f"connect {i+1}: {e}")
            import time; time.sleep(8)
    else:
        sys.exit(1)
    _, o, e = c.exec_command(f"bash -s << 'EOF'\n{SCRIPT}\nEOF", timeout=120)
    print(o.read().decode("utf-8", "replace"))
    err = e.read().decode("utf-8", "replace")
    if err.strip():
        print("STDERR:", err[-2000:])
    c.close()

if __name__ == "__main__":
    main()
