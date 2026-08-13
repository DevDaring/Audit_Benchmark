#!/usr/bin/env bash
# run_parallel.sh -- run the TIST battery over several models concurrently on one GPU.
#
# Why this is safe for the results.
#
#   Each model is an entirely independent computation. Running Llama and Qwen at the same
#   time does not change a single number in either: same weights, same prompts, same
#   batch size of one, same kernels. The processes share only VRAM. Every task already
#   writes to a per-model JSONL keyed by model name, so there is no shared output state
#   and no locking to get wrong.
#
#   What is deliberately NOT done here is batching several prompts into one forward pass.
#   That would be the other obvious speedup and it is the one that can move numbers:
#   padding changes reduction order and can select different kernels, so logits shift in
#   the last bits. The audit reports logit differences of order 0.5, and the placebo
#   comparison in E1.1 turns on separating small effects from smaller ones, so a
#   numerically perturbed forward pass is not worth the wall-clock. Batching stays off.
#
# VRAM budget on an 80 GB A100, bfloat16, measured from model sizes:
#
#   llama-3.1-8b-instruct   TransformerLens   ~32 GB  (HF copy plus HookedTransformer)
#   qwen2.5-7b-instruct     NNsight           ~16 GB  (wraps the HF model in place)
#   gemma-2-2b-it           TransformerLens   ~10 GB
#   phi-4-mini-instruct     NNsight            ~9 GB
#
#   TransformerLens holds a converted model alongside the loaded HF weights, so the two
#   TL models cost roughly double their parameter count. Models are launched largest
#   first, so the heavy pair overlaps with the light pair rather than two heavy models
#   landing together.
#
# Concurrency defaults to 2: peak resident is llama + qwen at about 48 GB, leaving ample
# headroom for activations and for the transient spike while TransformerLens converts a
# model. Raise it with TIST_PARALLEL=3 (about 58 GB peak) if the card is idle enough to
# justify it. Starts are staggered so two conversions never overlap.
#
# Usage:
#   bash TIST/run_parallel.sh                       # all tasks, all models, pool of 2
#   TIST_PARALLEL=3 bash TIST/run_parallel.sh
#   bash TIST/run_parallel.sh --dry                 # 2 units per task per model
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT="$(cd "$HERE/.." && pwd)"
# Default 1. This provider allocates the GPU as one indivisible unit and kills a second
# process that attaches to it: dry attempts at pool 3 and pool 2 both died with SIGTERM
# and no Python traceback, while sequential passed. Raise only on a provider known to
# permit GPU sharing, with TIST_PARALLEL=2.
#
# This default must stay in step with the `--parallel` default in run_tist_gpu.py. Having
# two knobs is what let the main run launch two processes after the dry run had already
# been fixed to run sequentially.
POOL="${TIST_PARALLEL:-1}"
STAGGER="${TIST_STAGGER:-120}"
EXTRA=("$@")

# Largest first. The pool then pairs a heavy model with a light one.
MODELS=(llama-3.1-8b-instruct qwen2.5-7b-instruct gemma-2-2b-it phi-4-mini-instruct)

mkdir -p "$HERE/logs"
cd "$AUDIT"

# Reduce fragmentation across long-lived allocations from concurrent processes.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[parallel] pool=$POOL stagger=${STAGGER}s models=${#MODELS[@]} extra=${EXTRA[*]:-none}"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

declare -a PIDS=()
declare -a PIDMODEL=()
FAILED=0

launch() {
  local m="$1"
  echo "[parallel] start $m"
  python3 TIST/run_tist_gpu.py --tasks all --models "$m" "${EXTRA[@]}" \
    > "$HERE/logs/run_${m}.log" 2>&1 &
  PIDS+=($!)
  PIDMODEL+=("$m")
}

# Wait for any one job to finish, reap it, and record failure.
reap_one() {
  local i pid m
  while true; do
    for i in "${!PIDS[@]}"; do
      pid="${PIDS[$i]}"
      [ -z "$pid" ] && continue
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; local rc=$?
        m="${PIDMODEL[$i]}"
        if [ "$rc" -ne 0 ]; then
          echo "[parallel] $m FAILED rc=$rc (see logs/run_${m}.log)"
          FAILED=$((FAILED+1))
        else
          echo "[parallel] $m done"
        fi
        PIDS[$i]=""
        return 0
      fi
    done
    sleep 15
  done
}

running() {
  local n=0 pid
  for pid in "${PIDS[@]}"; do
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && n=$((n+1))
  done
  echo "$n"
}

for m in "${MODELS[@]}"; do
  while [ "$(running)" -ge "$POOL" ]; do
    reap_one
  done
  launch "$m"
  sleep "$STAGGER"      # keep two TransformerLens conversions from overlapping
done

while [ "$(running)" -gt 0 ]; do
  reap_one
done

echo "[parallel] all models finished, failures=$FAILED"
nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true

# Non-zero only if every model failed; a partial run still carries usable results and the
# analysis reports per-model coverage.
if [ "$FAILED" -ge "${#MODELS[@]}" ]; then
  echo "[parallel] every model failed"
  exit 4
fi
exit 0
