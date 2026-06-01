"""
Full dataset rebuild on VM: stop GPU, upload fixes, rebuild pentad with validation.
"""
import sys
import io
import paramiko
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HOST, PORT, PW = "provider.a100.dsm.val.akash.pub", 31532, "MirageVM2026!"
REPO = "/data/Audit_Benchmark"
MIRAGE = f"{REPO}/Code/mirage"
LOCAL = Path(r"d:\PhD\Audit_Benchmark")

UPLOAD = [
    (LOCAL / "Code/mirage/Dataset/pentad_generator.py", f"{MIRAGE}/Dataset/pentad_generator.py"),
    (LOCAL / "Code/mirage/Dataset/validate_pentad.py", f"{MIRAGE}/Dataset/validate_pentad.py"),
    (LOCAL / "Code/mirage/Dataset/equivalence_sets.yaml", f"{MIRAGE}/Dataset/equivalence_sets.yaml"),
    (LOCAL / "Code/mirage/Dataset/sample_seeds.py", f"{MIRAGE}/Dataset/sample_seeds.py"),
    (LOCAL / "Code/mirage/Dataset/cot_attack_generator.py", f"{MIRAGE}/Dataset/cot_attack_generator.py"),
    (LOCAL / "Code/mirage/Dataset/context_shift_drafter.py", f"{MIRAGE}/Dataset/context_shift_drafter.py"),
    (LOCAL / "Code/mirage/patch_det_slots.py", f"{MIRAGE}/patch_det_slots.py"),
    (LOCAL / "Code/mirage/regenerate_api_slots.py", f"{MIRAGE}/regenerate_api_slots.py"),
    (LOCAL / "Code/mirage/run_dataset.py", f"{MIRAGE}/run_dataset.py"),
    (LOCAL / "Code/mirage/CPU_Only/scoring.py", f"{MIRAGE}/CPU_Only/scoring.py"),
    (LOCAL / "Code/mirage/CPU_Only/predictive_validity.py", f"{MIRAGE}/CPU_Only/predictive_validity.py"),
    (LOCAL / "Code/mirage/GPU_CPU/osm_behavioral.py", f"{MIRAGE}/GPU_CPU/osm_behavioral.py"),
    (LOCAL / "akash/_full_pipeline.py", f"{REPO}/akash/_full_pipeline.py"),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PW, timeout=25)
print("Connected.\n")

sftp = c.open_sftp()
for local, remote in UPLOAD:
    sftp.put(str(local), remote)
    print(f"  uploaded {remote}")
sftp.close()

REBUILD = f"""
set -uo pipefail
echo "=== Stop all pipeline processes ==="
pkill -f run_gpu_pipeline.py 2>/dev/null || true
pkill -f regenerate_api_slots 2>/dev/null || true
pkill -f _full_pipeline.py 2>/dev/null || true
pkill -f supervise_pipeline 2>/dev/null || true
sleep 2

echo "=== Clear state markers and partial GPU outputs ==="
rm -f /data/state/DATASET_OK /data/state/GPU_PIPELINE_OK /data/state/PIPELINE_COMPLETE
rm -f {MIRAGE}/Dataset/seeds/cot_attack_checkpoint.json
rm -f {MIRAGE}/Dataset/seeds/context_shift_checkpoint.json
rm -f {MIRAGE}/results/behavioral_results.parquet
rm -f {MIRAGE}/results/cdva_results.parquet

echo "=== Step 1: patch deterministic slots ==="
/data/venv/bin/python {MIRAGE}/patch_det_slots.py
PATCH_RC=$?
echo "patch_det_slots exit: $PATCH_RC"

if [ $PATCH_RC -ne 0 ]; then
  echo "PATCH FAILED — aborting"
  exit 1
fi

echo "=== Step 2: regenerate DeepSeek d/e slots ==="
nohup /data/venv/bin/python {MIRAGE}/regenerate_api_slots.py >> /data/logs/regenerate_api_slots.log 2>&1 &
echo "regen pid=$!"
echo "Monitor: tail -f /data/logs/regenerate_api_slots.log"
"""

_, o, e = c.exec_command(f"bash -s << 'EOF'\n{REBUILD}\nEOF", timeout=120)
print(o.read().decode("utf-8", "replace"))
err = e.read().decode("utf-8", "replace")
if err.strip():
    print("STDERR:", err[:500])
c.close()
