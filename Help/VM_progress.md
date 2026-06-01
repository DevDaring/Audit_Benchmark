# VM Progress — MIRAGE Akash Pipeline

Reference for monitoring, stage markers, ETAs, and safe resume on the production Akash GPU VM.

See also: **`Help/Akash_VM_Setup.md`** (deployment and troubleshooting) and **`Code/mirage/README.md`** (local dataset build).

---

## Current Production VM

Connection details live in `akash/vm_ssh.txt` (gitignored). Typical layout:

| Field | Value |
|---|---|
| SSH | `ssh root@<provider-host> -p <port>` |
| Password | `MirageVM2026!` |
| Repo | `/data/Audit_Benchmark` |
| Venv | `/data/venv` |
| HF cache | `/data/hf_cache` |
| Markers | `/data/state/` |
| Logs | `/data/logs/` |

---

## Pipeline Stages

```
INSTALL_OK        → akash/install.sh (venv + torch + flash_attn)
PREDOWNLOAD_OK    → akash/predownload_models.py (4 OSM models, no GPU)
DATASET_OK        → pentad validated (7,152 rows, assert_production_ready)
GPU_PIPELINE_OK   → behavioral + CDVA + tau calibration
PIPELINE_COMPLETE → final sentinel
```

### Stage detail

| Stage | What runs | Expected output |
|---|---|---|
| Install | `install.sh` | `/data/venv`, `INSTALL_OK` |
| Pre-download | `predownload_models.py` | ~42 GB in `/data/hf_cache`, `PREDOWNLOAD_OK` |
| Dataset — det | `patch_slot_b_only.py` or skip if valid | 4,172 rows (a/b/c only) |
| Dataset — API | `regenerate_api_slots.py` | +2,980 rows (d/e), total 7,152 |
| Dataset — gate | `assert_production_ready()` + manifest | `DATASET_OK`, `pentad_manifest.json` |
| GPU Step 1 | Load 4 OSM models | ~42 GB VRAM |
| GPU Step 2 | Behavioral eval | `behavioral_results.parquet` |
| GPU Step 3 | CDVA patching | `cdva_results.parquet` |
| GPU Step 4 | Tau calibration | `tau_calibration.json` |

**Production audit set:** N = **596** seeds × 12 slots = **7,152** rows (BBQ 254, CrowS-Pairs 181, StereoSet 161; 22 StereoSet seeds excluded).

---

## Local Monitoring Scripts

Run from repo root:

```bash
# Full snapshot: markers, pentad rows, manifest, production gate, GPU, logs
python akash/_vm_progress.py

# Health audit with ETA (recommended)
python akash/_pipeline_health.py

# DeepSeek regen checkpoint progress
python akash/_regen_progress.py

# Poll until PIPELINE_COMPLETE
python akash/_monitor.py
```

### What `_vm_progress.py` checks

1. Marker files in `/data/state/`
2. Pentad row count and slot distribution
3. `pentad_manifest.json`
4. `assert_production_ready()` pass/fail
5. Result parquets in `Code/mirage/results/`
6. Latest pipeline log tail
7. Supervisor / regen process list
8. `nvidia-smi` GPU utilisation

---

## On-VM Quick Checks

```bash
# Markers
ls -la /data/state/
for f in INSTALL_OK PREDOWNLOAD_OK DATASET_OK GPU_PIPELINE_OK PIPELINE_COMPLETE; do
  [ -f /data/state/$f ] && echo OK:$f || echo MISSING:$f
done

# Pentad shape (expect 7152 rows when complete)
/data/venv/bin/python -c "
import pandas as pd
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
print('rows', len(df), 'seeds', df.seed_id.nunique())
print(df.slot.value_counts().to_dict())
"

# Production gate
cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python -c "
import pandas as pd
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar
df = pd.read_parquet('Dataset/seeds/pentad_dataset.parquet')
validate_slot_b_grammar(df)
assert_production_ready(df)
print('PRODUCTION READY')
"

# Active processes
pgrep -af regenerate_api_slots
pgrep -af supervise_pipeline
pgrep -af run_gpu_pipeline
pgrep -af autonomous_guard

# Logs
tail -20 /data/logs/pipeline_attempt_1.log
tail -20 /data/Audit_Benchmark/LOG/regen_api_slots.log
tail -5 /data/logs/watchdog.log
```

---

## Expected Runtime & ETA

| Phase | Duration |
|---|---|
| Install (first time) | ~2–3 min |
| Pre-download (~42 GB) | ~5–10 min |
| DeepSeek regen (596 seeds, 2 parallel workers) | ~30–90 min |
| GPU behavioral (4 models × 7,152 prompts) | ~4–5 hr |
| CDVA + tau calibration | ~1–2 hr |
| **Total after dataset ready** | **~6 hr** |

During regen, checkpoint files report progress:

- `Dataset/seeds/context_shift_checkpoint.json` — slot-d (596 seeds)
- `Dataset/seeds/cot_attack_checkpoint.json` — slot-e (596 seeds)

At ~50–60 seeds/min with two DeepSeek keys, slot-d takes ~10–15 min; slot-e ~15–20 min.

During GPU behavioral, grep the log for throughput:

```bash
grep "prompts done" /data/logs/pipeline_attempt_1.log | tail -5
```

Observed rate on A100: ~147 prompts/min → ~6 hr for full GPU phase.

---

## Autonomous Guard (On-VM)

`akash/autonomous_guard.sh` runs on the VM during dataset rebuild:

- Polls every 60 s while `regenerate_api_slots.py` is active
- Restarts dead regen with `--keep-checkpoint`
- On `assert_production_ready()` pass → writes validation state → starts `supervise_pipeline.sh`
- Keeps supervisor **off** until the pentad is complete

Start manually after deploying fixes:

```bash
nohup bash /data/Audit_Benchmark/akash/autonomous_guard.sh \
  >> /data/logs/autonomous_guard.log 2>&1 &
```

---

## Safe Resume Rules

| Pentad state | Action |
|---|---|
| Det valid (4,172 rows), d/e missing | Run `regenerate_api_slots.py --keep-checkpoint`; do **not** run `patch_det_slots.py` |
| Slot-b grammar fails | `patch_slot_b_only.py` then regen if slot-a text changed |
| Full pentad valid, GPU interrupted | Clear `GPU_PIPELINE_OK` + `PIPELINE_COMPLETE` only; restart supervisor |
| Pentad SHA changed | Clear `DATASET_OK` + GPU markers; rebuild d/e |

**Always kill the supervisor before patching the pentad:**

```bash
pkill -f supervise_pipeline
pkill -f _full_pipeline
```

**Never** start GPU on a det-only pentad (4,172 rows without d/e slots).

---

## Deploy Code Fixes to Running VM

```bash
python akash/_deploy_hardened.py    # upload fixes + restart regen + start guard
python akash/_upload_mirage_fixes.py # upload Code/mirage only
```

Git pull on the VM is **disabled by default** (`MIRAGE_GIT_PULL=0`). Set `MIRAGE_GIT_PULL=1` only when you intentionally want to sync from GitHub.

---

## Post-Completion

When `PIPELINE_COMPLETE` is set:

1. Download `results/behavioral_results.parquet`, `cdva_results.parquet`, `tau_calibration.json`
2. Download `Dataset/seeds/pentad_dataset.parquet` and `pentad_manifest.json`
3. Run `CPU_Only/` scoring locally (see `Help/Akash_VM_Setup.md` §18)
4. Close the Akash deployment to stop billing
