#!/usr/bin/env bash
# P1e: pipeline supervisor — re-launches _full_pipeline.py on crash/eviction.
# Runs on every container boot (launched by SDL startup command).
# Resumes from /data/state markers — never re-does completed steps.
set -uo pipefail

REPO="${REPO_DIR:-/data/Audit_Benchmark}"
STATE="${STATE_DIR:-/data/state}"
VENV_PY="${VENV:-/data/venv}/bin/python"
mkdir -p /data/logs "$STATE"

echo "[supervise] $(date -u +%FT%TZ) boot — pid1_age=$(ps -p 1 -o etimes= 2>/dev/null | tr -d ' ')s"

# ── Already done? ─────────────────────────────────────────────────────────
if [ -f "$STATE/PIPELINE_COMPLETE" ]; then
  echo "[supervise] PIPELINE_COMPLETE already — nothing to do."
  exit 0
fi

# ── Wait for .env (uploaded via SFTP by _deploy_mirage.py) ───────────────
echo "[supervise] Waiting for /data/.env (HF_TOKEN needed for gated models)..."
WAITED=0
until [ -f /data/.env ] || [ "$WAITED" -ge 600 ]; do
  sleep 10; WAITED=$((WAITED + 10))
  [ $((WAITED % 60)) -eq 0 ] && echo "[supervise] still waiting for .env... ${WAITED}s"
done
if [ -f /data/.env ]; then
  echo "[supervise] .env found after ${WAITED}s — sourcing tokens"
  # Export HF_TOKEN from .env so predownload and TransformerLens can use it.
  # Use tr -d '\r' to strip Windows CRLF endings from the token value.
  HF_RAW=$(grep '^HUGGINGFACE_TOKEN=' /data/.env 2>/dev/null | head -1 || true)
  if [ -n "$HF_RAW" ]; then
    HF_TOKEN_CLEAN=$(printf '%s' "${HF_RAW#*=}" | tr -d '\r\n ')
    export HF_TOKEN="$HF_TOKEN_CLEAN"
    export HUGGINGFACE_TOKEN="$HF_TOKEN_CLEAN"
    echo "[supervise] HF_TOKEN exported (len=${#HF_TOKEN_CLEAN})"
  fi
  # Also export all keys from .env into the environment for Python subprocesses,
  # stripping \r from every value to handle Windows CRLF .env files.
  while IFS= read -r line; do
    line=$(printf '%s' "$line" | tr -d '\r')
    case "$line" in
      ''|\#*) continue ;;
      *=*)
        key="${line%%=*}"
        val="${line#*=}"
        export "$key"="$val"
        ;;
    esac
  done < /data/.env
else
  echo "[supervise] WARN: .env not found after 600s — gated model downloads may fail"
fi

# ── Pull latest code before first attempt (opt-in — uploaded fixes must not be overwritten) ──
if [ "${MIRAGE_GIT_PULL:-0}" = "1" ]; then
  git -C "$REPO" pull --ff-only origin main 2>&1 || true
else
  echo "[supervise] git pull skipped (set MIRAGE_GIT_PULL=1 to enable)"
fi

# ── Retry loop ─────────────────────────────────────────────────────────────
ATTEMPT=0
until [ -f "$STATE/PIPELINE_COMPLETE" ]; do
  ATTEMPT=$((ATTEMPT + 1))
  TS=$(date -u +%FT%TZ)
  AGE=$(ps -p 1 -o etimes= 2>/dev/null | tr -d ' ')
  LOG="/data/logs/pipeline_attempt_${ATTEMPT}.log"

  echo "[supervise] attempt $ATTEMPT at $TS (pid1_age=${AGE}s) → $LOG"

  # Use venv python if available, fall back to system python3
  PY="$VENV_PY"
  [ -x "$PY" ] || PY="python3"

  "$PY" "$REPO/akash/_full_pipeline.py" 2>&1 | tee -a "$LOG"

  if [ -f "$STATE/PIPELINE_COMPLETE" ]; then
    echo "[supervise] PIPELINE_COMPLETE after attempt $ATTEMPT"
    break
  fi

  echo "[supervise] attempt $ATTEMPT exited without completion — resuming in 15s"
  sleep 15
  if [ "${MIRAGE_GIT_PULL:-0}" = "1" ]; then
    git -C "$REPO" pull --ff-only origin main 2>&1 || true
  fi
done

echo "[supervise] done at $(date -u +%FT%TZ)"
