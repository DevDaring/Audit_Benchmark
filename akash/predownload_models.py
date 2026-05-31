"""
Pre-download all 4 OSM models to HuggingFace disk cache.

Called from install.sh after all Python packages are installed.
Reads HUGGINGFACE_TOKEN from the .env file that was staged at
/workspace/Audit_Benchmark/Code/mirage/.env before install.sh ran.

Why this exists:
  Each model download takes ~77 seconds over the Akash provider's
  network.  The dry run visits 4 models sequentially, so without a
  warm cache the download phase alone takes ~308 s.  Combined with
  install.sh (~210 s) the pipeline takes ~518+ s — close to the
  provider's container lifecycle window.

  Pre-downloading here means the dry run just loads models from disk
  (~5 s each), cutting total pipeline time to ~260 s and giving a
  comfortable margin.
"""
import os, sys
from pathlib import Path

ENV_FILE = Path("/workspace/Audit_Benchmark/Code/mirage/.env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
if not HF_TOKEN:
    print("[predownload] WARNING: HUGGINGFACE_TOKEN not found — models will download on first use", flush=True)
    sys.exit(0)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[predownload] WARNING: huggingface_hub not available — skipping", flush=True)
    sys.exit(0)

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-2b-it",
    "microsoft/Phi-4-mini-instruct",
]

for model_id in MODELS:
    print(f"[predownload] {model_id} ...", flush=True)
    try:
        snapshot_download(repo_id=model_id, token=HF_TOKEN)
        print(f"[predownload] OK: {model_id}", flush=True)
    except Exception as exc:
        print(f"[predownload] WARN: {model_id}: {exc}", flush=True)

print("[predownload] Done.", flush=True)
