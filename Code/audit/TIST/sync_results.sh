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
git -C "$REPO" config user.email "${GIT_AUTHOR_EMAIL:-mirage-audit@users.noreply.github.com}" 2>/dev/null || true
git -C "$REPO" config pull.rebase true 2>/dev/null || true

# Count result records, so the commit message says what was actually shipped.
count_records() {
  local n
  n=$(cat "$AUDIT"/results/tist/e1/*.jsonl "$AUDIT"/results/tist/e4/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "${n:-0}"
}

# One writer at a time. The bootstrap's push_results and this daemon both run git in the
# same working tree, and `git stash` blocking on another process's index.lock hangs
# without a message. That is the most likely reason lease 4's bootstrap went silent while
# the container stayed alive: not a crash, a wedge on a lock. Every git sequence in the
# repo now takes this mutex, and waits rather than racing.
GIT_LOCK="${GIT_LOCK:-/tmp/tist-git.lock}"

push_once() {
  if command -v flock >/dev/null 2>&1; then
    flock -w 600 "$GIT_LOCK" bash -c "$(declare -f _push_once_body count_records log); \
      REPO='$REPO' AUDIT='$AUDIT' REMOTE='$REMOTE' LOG='$LOG' _push_once_body"
    return $?
  fi
  _push_once_body
}

_push_once_body() {
  local n msg last
  n=$(count_records)

  # Only commit when the record count actually moved.
  #
  # The previous version rewrote SYNC_STATUS.txt with a fresh timestamp every cycle, so
  # `git diff --cached` was never empty and the daemon committed on every interval whether
  # or not anything had been computed. Commit age therefore carried no information, and the
  # external stall alarm reported OK four times running through a process that had been
  # hung for 53 minutes. Silence has to mean "no progress" for a liveness check to be worth
  # having.
  # Progress is "records changed OR a worker log advanced". Records alone is not enough:
  # the behavioural pass writes no JSONL until a language completes, so a records-only
  # rule suppressed the commit for the whole pass and took the logs down with it, leaving
  # no way to tell a working run from a hung one. Skipping only when BOTH are static
  # keeps silence meaningful while preserving visibility.
  local sig sigfile
  sigfile="${LAST_COUNT_FILE:-/tmp/tist-last-record-count}"
  sig="${n}:$(cat "$HERE"/logs/run_*.log "$HERE"/logs/main.log 2>/dev/null | wc -l | tr -d ' ')"
  last=$(cat "$sigfile" 2>/dev/null || echo "-1")
  if [ "$sig" = "$last" ]; then
    log "no new records and no log activity (${n}); skipping the commit"
    return 0
  fi
  echo "$sig" > "$sigfile"

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
