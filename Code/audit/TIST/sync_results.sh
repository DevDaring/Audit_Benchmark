#!/usr/bin/env bash
# sync_results.sh -- push results from the GPU VM to GitHub on a fixed interval.
#
# Runs as a standalone daemon beside the battery so a lost or reaped lease never costs
# more than one interval of work. The bootstrap starts it once the dry run has produced
# records; it can also be started by hand on any VM holding the repo.
#
# Usage:
#   bash TIST/sync_results.sh &                 # 15 minute interval (default)
#   INTERVAL=300 bash TIST/sync_results.sh &    # 5 minute interval
#   bash TIST/sync_results.sh --once            # single push, for use at shutdown
#
# Environment:
#   Github_Classic_Token   required for authenticated push; read from env or ../.env
#   INTERVAL               seconds between pushes, default 900
#   REPO                   repo root, default inferred from this file's location
#
# Design notes.
#
#   * results/ is gitignored, so every add is forced. That is deliberate: the working
#     copy stays clean for local development while the VM still ships its outputs.
#   * The runner and a human can both push to main. A rejected push is normal, not an
#     error, so the script rebases onto origin and retries rather than failing.
#   * Result files are force-added and would collide with a rebase, so they are stashed
#     across it and restored afterwards.
#   * Nothing here ever runs `git push --force`. A lost race costs one interval.
#   * Failures are logged and the loop continues. A sync daemon that exits on the first
#     network blip is worse than none, because it looks alive.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../.." && pwd)}"
AUDIT="$REPO/Code/audit"
INTERVAL="${INTERVAL:-900}"
LOG="$HERE/logs/sync_results.log"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

mkdir -p "$HERE/logs" "$AUDIT/results/tist"

log() { echo "[sync $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# Read the token from the environment, falling back to the .env the bootstrap wrote.
TOKEN="${Github_Classic_Token:-}"
if [ -z "$TOKEN" ] && [ -f "$AUDIT/.env" ]; then
  TOKEN=$(grep -E '^Github_Classic_Token=' "$AUDIT/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"' \r')
fi
if [ -z "$TOKEN" ]; then
  log "FATAL no Github_Classic_Token in environment or $AUDIT/.env"
  exit 1
fi
REMOTE="https://${TOKEN}@github.com/DevDaring/Audit_Benchmark.git"

git config --global --add safe.directory "$REPO" 2>/dev/null || true
git -C "$REPO" config user.name  "MIRAGE TIST Runner" 2>/dev/null || true
git -C "$REPO" config user.email "koushikdeb2009@gmail.com" 2>/dev/null || true
git -C "$REPO" config pull.rebase true 2>/dev/null || true

# Count result records, so the commit message says what was actually shipped.
count_records() {
  local n
  n=$(cat "$AUDIT"/results/tist/e1/*.jsonl "$AUDIT"/results/tist/e4/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

push_once() {
  local n msg
  n=$(count_records)
  msg="tist-sync: ${n} records @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "$msg" > "$AUDIT/results/tist/SYNC_STATUS.txt"

  git -C "$REPO" add -f \
    Code/audit/results/tist \
    Code/audit/TIST/logs >/dev/null 2>&1 || true

  if git -C "$REPO" diff --cached --quiet 2>/dev/null; then
    log "nothing new (${n} records)"
    return 0
  fi

  git -C "$REPO" commit -q -m "$msg" >/dev/null 2>&1 || true

  # Up to three attempts: a rejected push means someone else pushed first, which is
  # expected when a human is also committing to main.
  local attempt
  for attempt in 1 2 3; do
    if git -C "$REPO" push -q "$REMOTE" main >/dev/null 2>&1; then
      log "pushed ${n} records (attempt ${attempt})"
      return 0
    fi
    log "push rejected, rebasing onto origin (attempt ${attempt})"
    git -C "$REPO" stash -q --include-untracked >/dev/null 2>&1 || true
    git -C "$REPO" fetch -q "$REMOTE" main >/dev/null 2>&1 || true
    git -C "$REPO" rebase -q FETCH_HEAD >/dev/null 2>&1 || {
      git -C "$REPO" rebase --abort >/dev/null 2>&1 || true
      log "rebase failed; leaving the commit for the next interval"
    }
    git -C "$REPO" stash pop -q >/dev/null 2>&1 || true
    sleep 5
  done
  log "push still failing after 3 attempts; will retry next interval"
  return 1
}

if [ "$ONCE" -eq 1 ]; then
  push_once
  exit $?
fi

log "starting, interval ${INTERVAL}s, repo $REPO"
trap 'log "received termination signal; final push"; push_once; exit 0' TERM INT

while true; do
  push_once
  sleep "$INTERVAL"
done
