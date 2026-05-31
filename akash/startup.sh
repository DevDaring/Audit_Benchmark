#!/usr/bin/env bash
# =============================================================================
# startup.sh  — MIRAGE container entrypoint
# Runs inside the Akash GPU container at boot.
# Sequence:
#   1. Set up workspace directories
#   2. Clone / pull repo from GitHub
#   3. Write .env (injected via Akash env var MIRAGE_ENV_B64, a base64 blob)
#   4. Run install.sh to install all packages
#   5. Start SSH server so we can attach and monitor interactively
#   6. Drop into a tmux session ready for dry_run / GPU pipeline
# =============================================================================
set -euo pipefail
LOG=/workspace/startup.log
mkdir -p /workspace
exec > >(tee -a "$LOG") 2>&1

echo "[startup] $(date)  START"
echo "[startup] Container hostname: $(hostname)"
echo "[startup] Python: $(python3 --version)"
echo "[startup] nvcc:   $(nvcc --version 2>/dev/null | head -1 || echo 'nvcc not on PATH')"

# ------------------------------------------------------------------ workspace
WORKSPACE=/workspace
REPO_DIR="$WORKSPACE/Audit_Benchmark"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/DevDaring/Audit_Benchmark.git}"

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

# ------------------------------------------------------------------ clone repo
if [ -d "$REPO_DIR/.git" ]; then
    echo "[startup] Repo exists, pulling latest ..."
    git -C "$REPO_DIR" pull --ff-only
else
    echo "[startup] Cloning $GITHUB_REPO ..."
    git clone "$GITHUB_REPO" "$REPO_DIR"
fi

# ------------------------------------------------------------------ folder structure
# Matches the project layout expected by config.py and all run scripts.
mkdir -p "$REPO_DIR/Code/mirage/Results"
mkdir -p "$REPO_DIR/Code/mirage/Datasets"
mkdir -p "$WORKSPACE/logs"

echo "[startup] Folder structure ready."

# ------------------------------------------------------------------ .env
# The .env file must NOT be in the git repo.  Pass its contents as a
# base64-encoded Akash environment variable: MIRAGE_ENV_B64
if [ -n "${MIRAGE_ENV_B64:-}" ]; then
    echo "[startup] Decoding .env from MIRAGE_ENV_B64 ..."
    echo "$MIRAGE_ENV_B64" | base64 -d > "$REPO_DIR/Code/mirage/.env"
    chmod 600 "$REPO_DIR/Code/mirage/.env"
    echo "[startup] .env written."
elif [ -f "$WORKSPACE/.env" ]; then
    echo "[startup] Using pre-uploaded .env from $WORKSPACE/.env"
    cp "$WORKSPACE/.env" "$REPO_DIR/Code/mirage/.env"
    chmod 600 "$REPO_DIR/Code/mirage/.env"
else
    echo "[startup] WARNING: No .env found. API calls will fail."
    echo "[startup] Upload one via: scp .env root@<IP>:/workspace/.env"
fi

# ------------------------------------------------------------------ HF cache location
# Keep model weights in /workspace/.hf_cache to survive container restarts if
# the provider mounts /workspace as a persistent volume.
export HF_HOME="$WORKSPACE/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME" >> "$REPO_DIR/Code/mirage/.env" 2>/dev/null || true

# ------------------------------------------------------------------ install packages
echo "[startup] Running install.sh ..."
bash "$REPO_DIR/akash/install.sh"
echo "[startup] install.sh complete."

# ------------------------------------------------------------------ SSH
# Start SSH so we can log in for interactive monitoring.
mkdir -p /run/sshd
# Allow root login with password (needed for Akash provider SSH forwarding)
sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
/usr/sbin/sshd
echo "[startup] SSH server started on port 22."

# ------------------------------------------------------------------ tmux session
# Create a persistent tmux session called 'mirage' so that experiments
# survive SSH disconnects.  The session is pre-populated with the project
# directory and runs the dry run by default; interrupt (Ctrl-C) to skip.
cd "$REPO_DIR/Code/mirage"

if ! tmux has-session -t mirage 2>/dev/null; then
    tmux new-session -d -s mirage -x 220 -y 50
    tmux send-keys -t mirage "cd $REPO_DIR/Code/mirage" Enter
    tmux send-keys -t mirage "source .env 2>/dev/null || true" Enter
    tmux send-keys -t mirage "export HF_HOME=$HF_HOME" Enter
    tmux send-keys -t mirage "echo '>>> MIRAGE VM READY  <<<'" Enter
    tmux send-keys -t mirage "echo 'Run:  python Dry_Run/dry_run_all.py'" Enter
    tmux send-keys -t mirage "echo '      python GPU_CPU/osm_behavioral.py'" Enter
fi

echo "[startup] tmux session 'mirage' ready."
echo "[startup] Attach: tmux attach -t mirage"
echo "[startup] $(date)  STARTUP COMPLETE"

# Keep container alive (Akash terminates it otherwise)
tail -f /dev/null
