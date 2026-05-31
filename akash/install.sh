#!/usr/bin/env bash
# =============================================================================
# install.sh  — MIRAGE GPU VM package installer
# Container: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
# Python:    3.10.12 (system)
# GPU:       NVIDIA A100 40 GB
# =============================================================================
set -euo pipefail
LOG=/workspace/install.log
exec > >(tee -a "$LOG") 2>&1

echo "[install] $(date)  START"

# ------------------------------------------------------------------ system deps
apt-get update -qq
apt-get install -y --no-install-recommends \
    curl git tmux wget ninja-build \
    build-essential g++ cmake \
    libssl-dev libffi-dev \
    openssh-server \
    python3-pip python3-dev \
    > /dev/null

echo "[install] System packages done"

# ------------------------------------------------------------------ pip baseline
python3 -m pip install --upgrade pip setuptools wheel packaging ninja
echo "[install] pip/setuptools/wheel done"

# ------------------------------------------------------------------ PyTorch 2.6 + CUDA 12.4
# Prebuilt wheel for cu124.  flash-attn 2.7.4.post1 has an official prebuilt
# wheel for torch2.6+cu12+cxx11abiFALSE (450 K downloads, most stable choice).
echo "[install] Installing PyTorch 2.6.0 + CUDA 12.4 ..."
python3 -m pip install \
    torch==2.6.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

python3 -c "import torch; print('[install] torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

# ------------------------------------------------------------------ flash-attn
# Strategy (layered, stops at first success):
#   1. Official prebuilt wheel  (torch2.6 + cu12 + cxx11abiFALSE + cp310)
#   2. pip install --no-build-isolation  (auto-downloads matching wheel, falls
#      back to source compilation only if no prebuilt exists)
#   3. Abort with clear message
#
# IMPORTANT: cxx11abiFALSE is correct for pip-installed PyTorch on Ubuntu 22.04.
#   Pip PyTorch wheels always use the pre-CXX11 ABI; flash-attn must match.
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "[install] Attempting flash-attn prebuilt wheel ..."
if python3 -m pip install "$FLASH_WHL" --no-deps; then
    echo "[install] flash-attn prebuilt wheel installed"
else
    echo "[install] Prebuilt wheel failed; trying pip install --no-build-isolation ..."
    if MAX_JOBS=4 python3 -m pip install flash-attn --no-build-isolation; then
        echo "[install] flash-attn built from source"
    else
        echo "[install] ERROR: flash-attn installation failed on both paths."
        echo "[install] Check: python version (need 3.10), CUDA toolkit on PATH, torch version match."
        exit 1
    fi
fi

python3 -c "import flash_attn; print('[install] flash_attn', flash_attn.__version__)"

# ------------------------------------------------------------------ HuggingFace stack
echo "[install] Installing HuggingFace stack ..."
python3 -m pip install \
    "transformers>=4.47.0" \
    "accelerate>=0.34.0" \
    "datasets>=2.20.0" \
    "huggingface_hub>=0.25.0" \
    "tokenizers>=0.20.0" \
    "safetensors>=0.4.0" \
    "sentencepiece"

# ------------------------------------------------------------------ TransformerLens 2.18.0
# We pin 2.18.0 (last v2 release) because our CDVA code uses the v2 API
# (run_with_cache, run_with_hooks, to_tokens).  v3.x is API-compatible via a
# compatibility shim, but 2.18.0 avoids any risk of shim regressions.
# Note: 2.18.0 is fine with torch 2.6 despite not explicitly requiring it.
echo "[install] Installing TransformerLens 2.18.0 ..."
python3 -m pip install "transformer_lens==2.18.0"
python3 -c "import transformer_lens; print('[install] transformer_lens', transformer_lens.__version__)"

# ------------------------------------------------------------------ nnsight 0.7.0
# v0.6 removed the v0.4 compatibility layer; v0.7 is the latest stable.
# Our code uses LanguageModel + .trace + .save() which is stable in 0.7.
echo "[install] Installing nnsight 0.7.0 ..."
python3 -m pip install "nnsight==0.7.0"
python3 -c "import nnsight; print('[install] nnsight', getattr(nnsight,'__version__','ok'))"

# ------------------------------------------------------------------ outlines (constrained decoding)
echo "[install] Installing outlines ..."
python3 -m pip install "outlines>=0.1.0"

# ------------------------------------------------------------------ API clients
echo "[install] Installing API clients ..."
python3 -m pip install \
    "openai>=1.35.0" \
    "google-generativeai>=0.8.0" \
    "mistralai>=1.0.0" \
    "boto3>=1.34.0"

# ------------------------------------------------------------------ data / stats
echo "[install] Installing data science packages ..."
python3 -m pip install \
    "pandas>=2.0.0" \
    "pyarrow>=14.0.0" \
    "numpy>=1.24.0" \
    "scipy>=1.11.0" \
    "pyyaml>=6.0" \
    "python-dotenv>=1.0.0" \
    "tqdm>=4.66.0" \
    "requests>=2.31.0" \
    "paramiko>=3.4.0"

# ------------------------------------------------------------------ final verification
echo ""
echo "[install] ===== PACKAGE VERIFICATION ====="
python3 - <<'PYCHECK'
import sys, importlib, torch

checks = [
    ("torch",             lambda m: m.__version__),
    ("flash_attn",        lambda m: m.__version__),
    ("transformer_lens",  lambda m: m.__version__),
    ("nnsight",           lambda m: getattr(m, "__version__", "ok")),
    ("outlines",          lambda m: getattr(m, "__version__", "ok")),
    ("transformers",      lambda m: m.__version__),
    ("accelerate",        lambda m: m.__version__),
    ("openai",            lambda m: m.__version__),
    ("google.generativeai", lambda m: getattr(m, "__version__", "ok")),
    ("mistralai",         lambda m: getattr(m, "__version__", "ok")),
    ("boto3",             lambda m: m.__version__),
    ("pandas",            lambda m: m.__version__),
    ("pyarrow",           lambda m: m.__version__),
    ("scipy",             lambda m: m.__version__),
]

all_ok = True
for pkg, ver_fn in checks:
    try:
        mod = importlib.import_module(pkg)
        print(f"  OK  {pkg:<30} {ver_fn(mod)}")
    except Exception as exc:
        print(f"  FAIL {pkg:<30} {exc}", file=sys.stderr)
        all_ok = False

print(f"\n  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:  {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

sys.exit(0 if all_ok else 1)
PYCHECK

echo "[install] $(date)  DONE"
