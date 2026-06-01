"""
Production pre-launch audit + safe VM clean.

Phases:
  1. Stop supervisor/GPU, upload code, patch slot-b, start regen via nohup if needed
  2. Poll until regen completes (local reconnect loop)
  3. Full validation, safe clean, set DATASET_OK, start supervisor

Keeps: /data/venv, /data/hf_cache, seeds, pentad, INSTALL_OK, PREDOWNLOAD_OK, MODEL_* markers
Clears: GPU markers, partial results, stale regen checkpoints, archived pipeline attempt logs

Usage:
    python akash/_prelaunch_audit_clean.py
    python akash/_prelaunch_audit_clean.py --no-launch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"
REPO = "/data/Audit_Benchmark"
MIRAGE = f"{REPO}/Code/mirage"

UPLOAD = [
    ("Code/mirage/Dataset/pentad_generator.py", f"{MIRAGE}/Dataset/pentad_generator.py"),
    ("Code/mirage/Dataset/validate_pentad.py", f"{MIRAGE}/Dataset/validate_pentad.py"),
    ("Code/mirage/patch_slot_b_only.py", f"{MIRAGE}/patch_slot_b_only.py"),
    ("Code/mirage/regenerate_api_slots.py", f"{MIRAGE}/regenerate_api_slots.py"),
    ("Code/mirage/patch_det_slots.py", f"{MIRAGE}/patch_det_slots.py"),
    ("akash/_full_pipeline.py", f"{REPO}/akash/_full_pipeline.py"),
]

PHASE1 = r"""
set -euo pipefail
PY=/data/venv/bin/python
MIRAGE=/data/Audit_Benchmark/Code/mirage
SEEDS=$MIRAGE/Dataset/seeds
LOG=/data/logs/prelaunch_regen.log

echo "=== PHASE 1: stop supervisor/gpu/regen ==="
pkill -f supervise_pipeline.sh 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f run_gpu_pipeline.py 2>/dev/null || true
pkill -f regenerate_api_slots.py 2>/dev/null || true
sleep 3

echo "=== patch slot-b ==="
cd $MIRAGE
$PY patch_slot_b_only.py

echo "=== pentad before regen ==="
NEED_REGEN=1
if $PY - <<'PY'
import pandas as pd
df = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
audit = df[df.seed_source.str.lower().isin(["bbq","crows_pairs","stereoset"])]
de = int(df.slot.isin(["d","e"]).sum())
need = audit.seed_id.nunique() * 5
print("rows", len(df), "d/e", de, "need", need)
import sys
sys.exit(0 if de >= need else 1)
PY
then
  NEED_REGEN=0
fi
if [ "$NEED_REGEN" -eq 1 ]; then
  rm -f $SEEDS/context_shift_checkpoint.json $SEEDS/cot_attack_checkpoint.json
  echo "=== starting regen (nohup) ==="
  nohup $PY regenerate_api_slots.py >> $LOG 2>&1 &
  echo "regen_pid=$!"
else
  echo "=== d/e already present — skip regen ==="
fi
"""

POLL = r"""
pgrep -f 'regenerate_api_slots.py' >/dev/null && echo RUNNING || echo DONE
/data/venv/bin/python - <<'PY'
import json, os, pandas as pd
p = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet"
df = pd.read_parquet(p)
audit = df[df.seed_source.str.lower().isin(["bbq","crows_pairs","stereoset"])]
de = int(df.slot.isin(["d","e"]).sum())
need = audit.seed_id.nunique() * 5
ckpt = "/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json"
n_ckpt = len(json.load(open(ckpt))) if os.path.exists(ckpt) else 0
print("rows", len(df), "de", de, "need", need, "ctx_ckpt", n_ckpt)
PY
tail -1 /data/logs/prelaunch_regen.log 2>/dev/null || true
"""

PHASE3 = r"""
set -euo pipefail
PY=/data/venv/bin/python
MIRAGE=/data/Audit_Benchmark/Code/mirage
STATE=/data/state
LOGS=/data/logs
SEEDS=$MIRAGE/Dataset/seeds
RESULTS=$MIRAGE/results
LAUNCH=__LAUNCH__

echo "=== PHASE 3: production validation ==="
$PY - <<'PY'
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
import pandas as pd
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar, write_pentad_manifest
from Dataset.gold_utils import is_scorable_gold

df = pd.read_parquet("/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet")
audit = df[df.seed_source.str.lower().isin(["bbq","crows_pairs","stereoset"])]
print("rows", len(df), "seeds", audit.seed_id.nunique())
print("slots", df.slot.value_counts().to_dict())

slot_a = audit[audit.slot=="a"]
for src in ["bbq","crows_pairs","stereoset"]:
    sub = slot_a[slot_a.seed_source.str.lower()==src]
    sc = sub.apply(lambda r: is_scorable_gold(r["gold_answer"], r["seed_source"]), axis=1).sum()
    if sc < len(sub):
        raise SystemExit(f"FAIL gold {src}: {sc}/{len(sub)}")

validate_slot_b_grammar(df)
assert_production_ready(df)
write_pentad_manifest(df)

flags = []
for sid, g in audit.groupby("seed_id"):
    t = str(g[g.slot=="b"].iloc[0]["prompt_text"]).lower()
    for bad in ("person and person", "person man", "context: person", "person are"):
        if bad in t:
            flags.append(sid)
            break
if flags:
    raise SystemExit(f"FAIL grammar flags: {flags[:8]}")

wino = df[df.seed_source.str.lower()=="winobias"]
if len(wino):
    raise SystemExit(f"FAIL winobias rows: {len(wino)}")

print("VALIDATION_OK")
PY

echo "=== safe clean ==="
rm -f $STATE/DATASET_OK $STATE/GPU_PIPELINE_OK $STATE/PIPELINE_COMPLETE
rm -f $RESULTS/behavioral_results.parquet $RESULTS/cdva_results.parquet $RESULTS/tau_calibration.json
rm -f $SEEDS/context_shift_checkpoint.json $SEEDS/cot_attack_checkpoint.json
ARCH=$LOGS/archive_prelaunch_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$ARCH"
mv $LOGS/pipeline_attempt_*.log "$ARCH/" 2>/dev/null || true
echo "archived logs -> $ARCH"

echo "=== code smoke test ==="
$PY - <<'PY'
import sys
sys.path.insert(0, "/data/Audit_Benchmark/Code/mirage")
from Dataset.pentad_generator import _replace_with_neutral_person, _polish_slot_b_text
assert _replace_with_neutral_person("The Muslim man is very BLANK", "Muslim") == "The person is very BLANK"
assert "People are" in _replace_with_neutral_person("Context: Gentlemen are BLANK", "Gentlemen")
print("code_smoke_ok")
PY

touch $STATE/DATASET_OK
ls -la $STATE/

if [ "$LAUNCH" = "1" ]; then
  echo "=== start supervisor ==="
  nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> $LOGS/supervise.log 2>&1 &
  sleep 8
  pgrep -af supervise_pipeline || true
  tail -6 $LOGS/supervise.log
  echo "PRELAUNCH_OK launched"
else
  echo "PRELAUNCH_OK no-launch"
fi
"""


def connect() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(10):
        try:
            c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30, banner_timeout=60)
            return c
        except Exception as exc:
            print(f"  connect attempt {attempt + 1}: {exc}")
            time.sleep(12)
    raise RuntimeError("Cannot connect to VM")


def run(c: paramiko.SSHClient, script: str, timeout: int = 600) -> tuple[int, str, str]:
    _, stdout, stderr = c.exec_command(f"bash -s << 'EOF'\n{script}\nEOF", timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parents[1]
    launch_flag = "0" if args.no_launch else "1"

    print("=== Upload latest code ===")
    c = connect()
    sftp = c.open_sftp()
    for local_rel, remote in UPLOAD:
        print(f"  {local_rel}")
        sftp.put(str(root / local_rel), remote)
    sftp.close()

    print("\n=== Phase 1: patch + regen ===")
    code, out, err = run(c, PHASE1, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err[-4000:])
    c.close()
    if code != 0:
        sys.exit(code)

    print("\n=== Phase 2: wait for regen (up to 2h) ===")
    deadline = time.time() + 7200
    while time.time() < deadline:
        try:
            c = connect()
            code, out, err = run(c, POLL, timeout=60)
            c.close()
        except Exception as exc:
            print(f"  poll error: {exc}")
            time.sleep(20)
            continue

        print(time.strftime("%H:%M:%S"), out.strip().replace("\n", " | "))
        if "DONE" in out:
            done = False
            for line in out.splitlines():
                if line.startswith("rows") and "de" in line:
                    parts = line.split()
                    try:
                        de = int(parts[parts.index("de") + 1])
                        need = int(parts[parts.index("need") + 1])
                        if de >= need and de > 0:
                            print("Regen complete.")
                            done = True
                            break
                    except (ValueError, IndexError):
                        pass
            if done:
                break
        time.sleep(45)
    else:
        print("TIMEOUT waiting for regen")
        sys.exit(1)

    print("\n=== Phase 3: validate + clean + launch ===")
    c = connect()
    phase3 = PHASE3.replace("__LAUNCH__", launch_flag)
    code, out, err = run(c, phase3, timeout=600)
    print(out)
    if err.strip():
        print("STDERR:", err[-6000:])
    c.close()

    if code != 0 or "PRELAUNCH_OK" not in out:
        print(f"FAILED (exit {code})")
        sys.exit(code or 1)
    print("\nVM is production-ready.")


if __name__ == "__main__":
    main()
