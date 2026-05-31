#!/usr/bin/env bash
# =============================================================================
# install.sh  — MIRAGE GPU VM package installer
# Installs into a PERSISTENT venv at $VENV (/data/venv) so packages survive
# container evictions without reinstalling.
#
# Idempotent: if torch + flash_attn already import from the venv, exits 0
# immediately (saves ~150s + ~800 MB download per eviction cycle).
# =============================================================================
set -euo pipefail

VENV="${VENV:-/data/venv}"
STATE_DIR="${STATE_DIR:-/data/state}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/pip_cache}"
LOG=/data/logs/install.log

mkdir -p /data/logs "$STATE_DIR" "$PIP_CACHE_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[install] $(date -u +%FT%TZ) START  venv=$VENV"

# ── Idempotent skip ───────────────────────────────────────────────────────
if [ -x "$VENV/bin/python" ]; then
  if "$VENV/bin/python" -c "import torch, flash_attn; print('[install] venv already has torch', torch.__version__, 'and flash_attn', flash_attn.__version__)" 2>/dev/null; then
    echo "[install] venv already fully populated — skipping install (saved ~150s)"
    touch "$STATE_DIR/INSTALL_OK"
    exit 0
  fi
fi

# ── System dependencies (fast — runs on ephemeral root, ~20s) ────────────
echo "[install] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    curl wget git tmux openssh-server \
    python3-dev python3-venv build-essential g++ cmake \
    ninja-build libssl-dev libffi-dev \
    > /dev/null 2>&1
rm -rf /var/lib/apt/lists/*
echo "[install] System packages done"

# ── Create/update venv (on persistent /data) ─────────────────────────────
echo "[install] Creating venv at $VENV ..."
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip setuptools wheel packaging ninja

# ── PyTorch 2.6.0 + CUDA 12.4 ────────────────────────────────────────────
echo "[install] Installing PyTorch 2.6.0 + CUDA 12.4..."
"$VENV/bin/pip" install \
    torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

"$VENV/bin/python" -c "import torch; print('[install] torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ── Flash Attention 2.7.4.post1 (prebuilt wheel — no compiler needed) ────
# cxx11abiFALSE matches pip-distributed PyTorch on Ubuntu 22.04.
# --no-deps prevents pip from replacing our pinned torch.
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "[install] Installing flash-attn 2.7.4.post1 prebuilt wheel..."
if "$VENV/bin/pip" install "$FLASH_WHL" --no-deps; then
    echo "[install] flash-attn prebuilt wheel installed"
else
    echo "[install] Prebuilt wheel failed; building from source (slow: ~45min)..."
    MAX_JOBS=4 "$VENV/bin/pip" install flash-attn --no-build-isolation
fi
"$VENV/bin/python" -c "import flash_attn; print('[install] flash_attn', flash_attn.__version__)"

# ── HuggingFace stack ─────────────────────────────────────────────────────
echo "[install] Installing HuggingFace stack..."
"$VENV/bin/pip" install \
    "transformers>=4.47.0" \
    "accelerate>=0.34.0" \
    "datasets>=2.20.0" \
    "huggingface_hub>=0.25.0" \
    "tokenizers>=0.20.0" \
    "safetensors>=0.4.0" \
    "sentencepiece" \
    "hf_transfer"      # faster HF downloads (HF_HUB_ENABLE_HF_TRANSFER=1)

# ── TransformerLens 2.18.0 (pin v2 — CDVA code uses v2 API) ──────────────
echo "[install] Installing TransformerLens 2.18.0..."
"$VENV/bin/pip" install "transformer_lens==2.18.0"
"$VENV/bin/python" -c "import transformer_lens; print('[install] transformer_lens OK (v2.18.0 has no __version__ attr)')"

# ── nnsight 0.7.0 ────────────────────────────────────────────────────────
echo "[install] Installing nnsight 0.7.0..."
"$VENV/bin/pip" install "nnsight==0.7.0"
"$VENV/bin/python" -c "import nnsight; print('[install] nnsight', getattr(nnsight,'__version__','ok'))"

# ── Constrained decoding ──────────────────────────────────────────────────
echo "[install] Installing outlines..."
"$VENV/bin/pip" install "outlines>=0.1.0"

# ── API clients ───────────────────────────────────────────────────────────
echo "[install] Installing API clients..."
"$VENV/bin/pip" install \
    "openai>=1.35.0" \
    "google-generativeai>=0.8.0" \
    "mistralai>=1.0.0" \
    "boto3>=1.34.0"

# ── Data / stats ──────────────────────────────────────────────────────────
echo "[install] Installing data science packages..."
"$VENV/bin/pip" install \
    "pandas>=2.0.0" \
    "pyarrow>=14.0.0" \
    "numpy>=1.24.0" \
    "scipy>=1.11.0" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" \
    "requests>=2.31.0" \
    "paramiko>=3.4.0"

# Explicit dotenv verification (catches silent pip install failures)
"$VENV/bin/python" -c "import dotenv" || "$VENV/bin/pip" install --force-reinstall "python-dotenv>=1.0.0"

# ── Final verification ────────────────────────────────────────────────────
echo ""
echo "[install] ===== PACKAGE VERIFICATION ====="
"$VENV/bin/python" - <<'PYCHECK'
import sys, importlib, torch

checks = [
    ("torch",             lambda m: m.__version__),
    ("flash_attn",        lambda m: m.__version__),
    ("transformer_lens",  lambda m: getattr(m, "__version__", "ok-v2.18")),
    ("nnsight",           lambda m: getattr(m, "__version__", "ok")),
    ("outlines",          lambda m: getattr(m, "__version__", "ok")),
    ("transformers",      lambda m: m.__version__),
    ("accelerate",        lambda m: m.__version__),
    ("openai",            lambda m: m.__version__),
    ("google.generativeai", lambda m: getattr(m, "__version__", "ok")),
    ("mistralai",         lambda m: getattr(m, "__version__", "ok")),
    ("boto3",             lambda m: m.__version__),
    ("pandas",            lambda m: m.__version__),
    ("scipy",             lambda m: m.__version__),
    ("dotenv",            lambda m: "ok"),
]

all_ok = True
for pkg, ver_fn in checks:
    try:
        mod = importlib.import_module(pkg)
        print(f"  OK  {pkg:<30} {ver_fn(mod)}")
    except Exception as exc:
        print(f"  FAIL {pkg:<30} {exc}", file=sys.stderr)
        all_ok = False

print(f"\n  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

sys.exit(0 if all_ok else 1)
PYCHECK

touch "$STATE_DIR/INSTALL_OK"
echo "[install] $(date -u +%FT%TZ) DONE"
