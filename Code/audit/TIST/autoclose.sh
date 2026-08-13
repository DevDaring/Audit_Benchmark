#!/usr/bin/env bash
# autoclose.sh -- watch for the GPU battery to finish, then destroy the Akash lease.
#
# An idle A100 bills about $3.75/hour, roughly $90 a day, and the container sleeps rather
# than exiting when the run completes. Destroying it should not depend on a person being
# awake, so this watches for completion and closes the deployment itself.
#
# Trigger: a `tist-gpu: COMPLETE` commit dated AFTER this watcher started.
#
# It deliberately does not key off the TIST_GPU_DONE file. That file is committed to the
# repository and a stale copy from an earlier lease is already present: lease 1 reported
# COMPLETE having computed nothing and left the marker behind. A file-presence trigger
# would have fired instantly and destroyed a healthy running lease. A dry run caught
# exactly that, which is why the trigger is a freshly dated commit instead.
#
# Before closing it checks that real results exist, so a completion marker without data
# leaves the lease alive for inspection rather than throwing the evidence away.
#
# Usage:
#   bash TIST/autoclose.sh              # poll every 5 minutes, close when done
#   POLL=60 bash TIST/autoclose.sh      # faster poll
#   bash TIST/autoclose.sh --dry-run    # report only, never close
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDIT="$(cd "$HERE/.." && pwd)"
POLL="${POLL:-300}"
API="https://api.github.com/repos/DevDaring/Audit_Benchmark/contents/Code/audit/results/tist"
COMMITS="https://api.github.com/repos/DevDaring/Audit_Benchmark/commits?per_page=10"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# Enough result bytes that a marker with no data cannot trigger a close.
MIN_RESULT_BYTES=200000

result_bytes() {
  local total=0 part
  for d in e1 e4; do
    part=$(curl -s "$API/$d" 2>/dev/null | python -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(sum(f['size'] for f in d) if isinstance(d, list) else 0)
except Exception:
    print(0)
" 2>/dev/null || echo 0)
    total=$((total + part))
  done
  echo "$total"
}

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

fresh_complete() {
  curl -s "$COMMITS" 2>/dev/null | START="$START_UTC" python -c "
import sys, json, os, datetime
start = datetime.datetime.fromisoformat(os.environ['START'].replace('Z', '+00:00'))
try:
    data = json.load(sys.stdin)
except Exception:
    print(0); raise SystemExit
for c in data:
    msg = c['commit']['message'].splitlines()[0]
    ts = datetime.datetime.fromisoformat(c['commit']['committer']['date'].replace('Z', '+00:00'))
    if 'COMPLETE' in msg and ts > start:
        print(1); raise SystemExit
print(0)
" 2>/dev/null || echo 0
}

echo "[autoclose] armed at ${START_UTC}, polling every ${POLL}s (dry-run=${DRY})"
echo "[autoclose] trigger: a 'tist-gpu: COMPLETE' commit dated after that instant"

while true; do
  if [ "$(fresh_complete)" = "1" ]; then
    bytes=$(result_bytes)
    echo "[autoclose] fresh COMPLETE commit seen; results total ${bytes} B"
    if [ "$bytes" -lt "$MIN_RESULT_BYTES" ]; then
      echo "[autoclose] REFUSING to close: only ${bytes} B of results, expected >= ${MIN_RESULT_BYTES} B."
      echo "[autoclose] Lease stays up for inspection. Close by hand when satisfied:"
      echo "[autoclose]   python TIST/deploy_tist.py --close"
      exit 2
    fi
    if [ "$DRY" -eq 1 ]; then
      echo "[autoclose] dry-run: would close the lease now"
      exit 0
    fi
    echo "[autoclose] closing the lease"
    cd "$AUDIT" || exit 1
    if python TIST/deploy_tist.py --close 2>&1 | tail -3; then
      echo "[autoclose] LEASE CLOSED. Billing has stopped."
    else
      echo "[autoclose] CLOSE FAILED. Close by hand: python TIST/deploy_tist.py --close"
      exit 3
    fi
    exit 0
  fi
  sleep "$POLL"
done
