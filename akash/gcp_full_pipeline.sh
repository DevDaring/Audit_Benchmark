#!/bin/bash
# Full MIRAGE pipeline on GCP — logs to ~/mirage-pipeline.log
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HOME}/hf_cache"
export STATE_DIR="${HOME}/mirage-state"
export MIRAGE_ROOT="${HOME}/Audit_Benchmark"
export MIRAGE_CODE="${MIRAGE_ROOT}/Code/mirage"
LOG="${HOME}/mirage-pipeline.log"

exec > >(tee -a "${LOG}") 2>&1
echo "=== MIRAGE GCP pipeline started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

cd "${MIRAGE_CODE}"
sed -i 's/\r$//' .env
sed -i '/^PHONE_NO/d;/^TextBelt/d' .env
set -a && source .env && set +a
export HF_TOKEN="${HUGGINGFACE_TOKEN}"

# --- Pre-download OSM weights (resume-safe) ---
if [[ ! -f "${STATE_DIR}/PREDOWNLOAD_OK" ]]; then
  echo "[1/7] Pre-downloading HuggingFace models ..."
  export STATE_DIR HF_HOME
  python3 "${MIRAGE_ROOT}/akash/predownload_models.py" || true
fi

# --- Source datasets ---
echo "[2/7] Downloading source benchmarks ..."
python3 -c "
from Dataset.download_bbq import download_bbq, validate_bbq
from Dataset.download_crows_pairs import download_crows_pairs, validate_crows_pairs
from Dataset.download_stereoset import download_stereoset, validate_stereoset
from Dataset.download_winobias import download_winobias, validate_winobias
validate_bbq(download_bbq())
validate_crows_pairs(download_crows_pairs())
validate_stereoset(download_stereoset())
validate_winobias(download_winobias())
print('datasets OK')
"

# --- Seeds + pentad ---
echo "[3/7] Building pentad (596 seeds x 12) ..."
python3 -c "
from Dataset.sample_seeds import sample_seeds, verify_seeds_integrity
main, dev = sample_seeds()
verify_seeds_integrity()
print(len(main), 'main', len(dev), 'dev')
"
python3 run_dataset.py
python3 patch_slot_b_only.py
python3 regenerate_api_slots.py --keep-checkpoint || python3 regenerate_api_slots.py

# --- Research validation gates ---
echo "[4/7] Production + PAV validation ..."
python3 -c "
import pandas as pd
from Dataset.validate_pentad import assert_production_ready, validate_slot_b_grammar, write_pentad_manifest
df = pd.read_parquet('Dataset/seeds/pentad_dataset.parquet')
validate_slot_b_grammar(df)
assert_production_ready(df)
write_pentad_manifest(df)
print('pentad gate OK', len(df), 'rows')
"
python3 pav_validate.py || echo "WARN: pav_validate returned non-zero (review log)"

# --- Dry run: 2 seeds ---
echo "[5/7] Dry run (n_seeds=2) ..."
python3 Dry_Run/dry_run_all.py --n-seeds 2

# --- Full GPU pipeline ---
echo "[6/7] GPU pipeline (sequential 40GB mode) ..."
export MIRAGE_SEQUENTIAL_MODELS=1
export MIRAGE_EVAL_BATCH_SIZE=4
python3 GPU_CPU/run_gpu_pipeline.py

echo "[7/7] DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "=== MIRAGE GCP pipeline complete ==="
