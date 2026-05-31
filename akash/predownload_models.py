"""
Pre-download all 4 OSM models to the PERSISTENT HuggingFace cache (/data/hf_cache).

Key design principles (from expert review):
  - Cache lives on the PERSISTENT volume (/data) — survives container evictions.
  - Per-model marker files in $STATE_DIR — each model is downloaded exactly once
    per lease (not per eviction cycle). If evicted mid-download, snapshot_download
    resumes automatically from the partially-downloaded blobs.
  - HF_HOME is set in the SDL env to /data/hf_cache so HuggingFace uses it
    automatically in all subsequent code (no hardcoding needed).
  - HF_TOKEN is read from /data/.env (for gated repos: LLaMA-3.1 and Gemma-2).
"""
import os, sys, pathlib

# ── Paths from SDL env vars (with fallbacks for manual runs) ─────────────
STATE    = pathlib.Path(os.environ.get("STATE_DIR", "/data/state"))
HF_HOME  = os.environ.get("HF_HOME", "/data/hf_cache")
STATE.mkdir(parents=True, exist_ok=True)

# ── Load tokens from .env (supervisor already sourced HF_TOKEN into env) ─
ENV_FILE = pathlib.Path("/data/.env")
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or ""
)

if not HF_TOKEN:
    print("[predownload] WARN: no HF_TOKEN — gated models (LLaMA, Gemma) will return 401", flush=True)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[predownload] WARN: huggingface_hub not installed — skipping", flush=True)
    sys.exit(0)

# ── Model list — must match config.py OSM_MODELS[*]["hf_id"] exactly ─────
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",   # ~16 GB bf16  (GATED — needs HF_TOKEN)
    "Qwen/Qwen2.5-7B-Instruct",            # ~14 GB bf16  (public)
    "google/gemma-2-2b-it",                # ~4 GB  bf16  (GATED — needs HF_TOKEN)
    "microsoft/Phi-4-mini-instruct",       # ~8 GB  bf16  (public)
]

def marker_name(repo_id: str) -> str:
    return f"MODEL_{repo_id.replace('/', '__')}_OK"

all_ok = True
for repo_id in MODELS:
    marker = STATE / marker_name(repo_id)
    if marker.exists():
        print(f"[predownload] {repo_id} already cached — skip", flush=True)
        continue

    print(f"[predownload] Downloading {repo_id} → {HF_HOME}", flush=True)
    try:
        # snapshot_download resumes partial downloads automatically via
        # .incomplete blob files. Eviction mid-download is safe.
        snapshot_download(
            repo_id=repo_id,
            cache_dir=HF_HOME,
            token=HF_TOKEN or None,
            max_workers=4,   # lower to 2 if watchdog still shows disk pressure
        )
        marker.touch()
        print(f"[predownload] OK {repo_id}", flush=True)
    except Exception as exc:
        print(f"[predownload] WARN {repo_id}: {exc}", flush=True)
        all_ok = False

if all_ok:
    (STATE / "PREDOWNLOAD_OK").touch()
    print("[predownload] PREDOWNLOAD_OK — all models cached", flush=True)
else:
    print("[predownload] Some models failed — supervisor will retry", flush=True)
    sys.exit(1)
