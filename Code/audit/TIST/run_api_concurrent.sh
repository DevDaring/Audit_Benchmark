#!/usr/bin/env bash
# run_api_concurrent.sh -- multilingual API runs, concurrent by PROVIDER, not by model.
#
# Safety of the concurrency, which is the whole design question here.
#
#   Disjoint outputs. Every (model, language) writes its own parquet, so two workers can
#   never touch the same file. There is no shared state and therefore no lock to get wrong
#   between providers.
#
#   Quota, not CPU, is the constraint. The work is I/O bound, so 32 cores are irrelevant;
#   what matters is that two models sharing one vendor account share its rate limit.
#   qwen3-next and nova both run on AWS Bedrock under one key pair, and the probe already
#   logged two ThrottlingExceptions from nova alone. They therefore run SEQUENTIALLY
#   inside a single worker. gemini (MegaLLM) and mistral (Mistral) have their own
#   endpoints and run as separate workers.
#
#   No duplicate calls. Each worker owns a fixed model list passed with --models, the
#   lists are disjoint, and the runner itself resumes by reading the existing parquet and
#   skipping prompt_ids already present. Three independent reasons the same prompt cannot
#   be evaluated twice.
#
# Expected wall clock, from the probe's measured per-model latency:
#   worker A  qwen3-next 2.3 h then nova 4.1 h  = 6.4 h
#   worker B  gemini                            = 4.9 h
#   worker C  mistral                           = 2.2 h
#   total = the slowest worker, about 6.5 h, against roughly 14 h sequential.
#
# Usage:
#   bash TIST/run_api_concurrent.sh              # all three workers
#   LANGS="hi" bash TIST/run_api_concurrent.sh   # one language
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT="$(cd "$HERE/.." && pwd)"
PY="${PY:-/home/Debz/Research/tmp/venv/bin/python}"
LOGS="${LOGS:-/home/Debz/Research/tmp}"
LANGS="${LANGS:-hi bn}"

cd "$AUDIT"
export PYTHONPATH="$AUDIT"
mkdir -p "$LOGS"

launch() {                     # launch <worker-name> <model> [model...]
  local name="$1"; shift
  local log="$LOGS/api_${name}.log"
  echo "[concurrent] worker ${name}: $* -> ${log}"
  # Models inside one worker run sequentially, which is what serialises the shared
  # Bedrock quota. The runner loops its --models list in order.
  nohup "$PY" TIST/e4_api_multilingual.py --models "$@" --langs $LANGS > "$log" 2>&1 &
  echo "$!" > "$LOGS/api_${name}.pid"
}

echo "[concurrent] langs: $LANGS"
launch bedrock  qwen3-next-80b-a3b amazon-nova-2-lite
sleep 5
launch gemini   gemini-2.5-flash
sleep 5
launch mistral  mistral-medium

sleep 10
echo "[concurrent] running workers:"
for w in bedrock gemini mistral; do
  pid=$(cat "$LOGS/api_${w}.pid" 2>/dev/null || echo "")
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "  ${w}: pid ${pid} alive"
  else
    echo "  ${w}: NOT RUNNING (check $LOGS/api_${w}.log)"
  fi
done
echo "[concurrent] follow with: tail -f $LOGS/api_*.log"
