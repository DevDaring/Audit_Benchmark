"""
On-VM checkpoint-driven pipeline orchestrator (production).

Called by akash/supervise_pipeline.sh on every container boot.
Each step checks /data/state/<MARKER> before running — already-done steps
are skipped instantly. Resumable across container evictions.

Production run order (GPU_CPU only — CPU_Only/ is NOT executed here):
  INSTALL_OK        → persistent venv + packages (idempotent)
  PREDOWNLOAD_OK    → all 4 OSM models → /data/hf_cache
  DATASET_OK        → pentad_dataset.parquet (API/CPU; no GPU)
  GPU_PIPELINE_OK   → behavioral + CDVA + tau cal (GPU_CPU/run_gpu_pipeline.py)
  PIPELINE_COMPLETE → all production steps succeeded

Dry run is skipped in production (validated separately).
Set MIRAGE_RUN_DRYRUN=1 to force a 2-seed dry run gate before GPU work.
"""
import logging
import os
import pathlib
import shutil
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [pipeline] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)

STATE = pathlib.Path(os.environ.get("STATE_DIR", "/data/state"))
REPO  = pathlib.Path(os.environ.get("REPO_DIR", "/data/Audit_Benchmark"))
VENV  = pathlib.Path(os.environ.get("VENV", "/data/venv"))
MIRAGE = REPO / "Code" / "mirage"
PENTAD = MIRAGE / "Dataset" / "seeds" / "pentad_dataset.parquet"

STATE.mkdir(parents=True, exist_ok=True)


def _venv_python() -> str:
    candidate = str(VENV / "bin" / "python")
    if pathlib.Path(candidate).exists():
        return candidate
    return sys.executable


def done(name: str) -> bool:
    return (STATE / name).exists()


def mark(name: str) -> None:
    (STATE / name).touch()
    log.info("MARKER written: %s", name)


def _base_env(extra: dict = None) -> dict:
    env = {**os.environ, **(extra or {})}
    env["PATH"] = str(VENV / "bin") + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(MIRAGE)
    env.setdefault("HF_HOME", "/data/hf_cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf_cache/hub")
    env.setdefault("PIP_CACHE_DIR", "/data/pip_cache")
    env.setdefault("XDG_CACHE_HOME", "/data/cache")
    return env


def step(marker: str, argv: list, cwd=None, extra_env: dict = None) -> None:
    if done(marker):
        log.info("SKIP %s (marker already present)", marker)
        return
    log.info("START %s: %s", marker, " ".join(str(a) for a in argv))
    r = subprocess.run(argv, cwd=str(cwd or REPO), env=_base_env(extra_env))
    if r.returncode != 0:
        log.error("FAILED %s (exit %d) — supervisor will retry", marker, r.returncode)
        sys.exit(r.returncode)
    mark(marker)


# ── Step 1: Install ───────────────────────────────────────────────────────
step("INSTALL_OK", ["bash", str(REPO / "akash" / "install.sh")])

PY = _venv_python()
log.info("Using Python: %s", PY)

# ── Step 2: Copy .env (strip CRLF in Python path via config load) ─────────
env_src  = pathlib.Path("/data/.env")
env_dest = MIRAGE / ".env"
if env_src.exists():
    raw = env_src.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    env_dest.write_bytes(raw)
    log.info("Copied /data/.env → %s (CRLF normalised)", env_dest)
elif not env_dest.exists():
    log.warning(".env missing — API keys may be absent")

# ── Step 3: Pre-download models ───────────────────────────────────────────
PY = _venv_python()
step("PREDOWNLOAD_OK", [PY, str(REPO / "akash" / "predownload_models.py")])

# ── Optional dry-run gate ───────────────────────────────────────────────────
if os.environ.get("MIRAGE_RUN_DRYRUN", "").strip() in ("1", "true", "yes"):
    PY = _venv_python()
    step(
        "DRYRUN_OK",
        [PY, str(MIRAGE / "Dry_Run" / "dry_run_gpu_cpu.py"), "--n-seeds", "2"],
        cwd=MIRAGE,
    )

# ── Step 4: Dataset build (CPU + DeepSeek API — NOT GPU, NOT CPU_Only/) ───
if PENTAD.exists() and not done("DATASET_OK"):
    log.info("pentad_dataset.parquet already present (%d bytes) — marking DATASET_OK",
             PENTAD.stat().st_size)
    mark("DATASET_OK")
else:
    PY = _venv_python()
    step("DATASET_OK", [PY, str(MIRAGE / "run_dataset.py")], cwd=MIRAGE)

# ── Step 5: Full GPU pipeline (GPU_CPU/ only) ─────────────────────────────
PY = _venv_python()
step(
    "GPU_PIPELINE_OK",
    [PY, str(MIRAGE / "GPU_CPU" / "run_gpu_pipeline.py")],
    cwd=MIRAGE,
)

mark("PIPELINE_COMPLETE")
log.info("=== PIPELINE_COMPLETE — production GPU pipeline finished ===")
