"""
On-VM checkpoint-driven pipeline orchestrator.

Called by akash/supervise_pipeline.sh on every container boot.
Each step checks /data/state/<MARKER> before running — already-done steps
are skipped instantly. This makes the full pipeline resumable across any
number of container evictions.

State markers survive because /data is a PERSISTENT Akash volume.

Run order:
  INSTALL_OK       → install Python packages into /data/venv (idempotent)
  PREDOWNLOAD_OK   → download all 4 models to /data/hf_cache  (resumable)
  DRYRUN_OK        → 2-seed GPU dry run loading from cache      (clean exit)
  PIPELINE_COMPLETE → written only when all steps succeed
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

# ── Paths (all on persistent /data where possible) ───────────────────────
STATE = pathlib.Path(os.environ.get("STATE_DIR", "/data/state"))
REPO  = pathlib.Path(os.environ.get("REPO_DIR", "/data/Audit_Benchmark"))
VENV  = pathlib.Path(os.environ.get("VENV", "/data/venv"))


def _venv_python() -> str:
    """Return the venv python path; re-evaluated after install creates the venv."""
    candidate = str(VENV / "bin" / "python")
    if pathlib.Path(candidate).exists():
        return candidate
    return sys.executable

STATE.mkdir(parents=True, exist_ok=True)


def done(name: str) -> bool:
    return (STATE / name).exists()


def mark(name: str) -> None:
    (STATE / name).touch()
    log.info("MARKER written: %s", name)


def step(marker: str, argv: list, cwd=None, extra_env: dict = None) -> None:
    """Run a pipeline step unless its marker already exists."""
    if done(marker):
        log.info("SKIP %s (marker already present)", marker)
        return
    log.info("START %s: %s", marker, " ".join(str(a) for a in argv))
    env = {**os.environ, **(extra_env or {})}
    # Ensure venv is on PATH so sub-scripts can call `python` directly
    env["PATH"] = str(VENV / "bin") + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = str(REPO / "Code" / "mirage")
    # All cache dirs already set in SDL env; re-assert here for manual runs
    env.setdefault("HF_HOME", "/data/hf_cache")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/data/hf_cache/hub")
    env.setdefault("PIP_CACHE_DIR", "/data/pip_cache")
    env.setdefault("XDG_CACHE_HOME", "/data/cache")

    r = subprocess.run(argv, cwd=str(cwd or REPO), env=env)
    if r.returncode != 0:
        log.error("FAILED %s (exit %d) — supervisor will retry this step", marker, r.returncode)
        sys.exit(r.returncode)
    mark(marker)


# ── Step 1: Install packages into persistent venv ─────────────────────────
step("INSTALL_OK", ["bash", str(REPO / "akash" / "install.sh")])

# After install, venv now exists — always use it for subsequent steps
PY = _venv_python()
log.info("Using Python: %s", PY)

# ── Step 2: Copy .env to repo runtime path ────────────────────────────────
# The supervisor uploaded /data/.env via SFTP before starting.
# Copy it to the path the codebase expects, but only if not already there.
env_src  = pathlib.Path("/data/.env")
env_dest = REPO / "Code" / "mirage" / ".env"
if env_src.exists():
    if not env_dest.exists() or env_src.stat().st_mtime > env_dest.stat().st_mtime:
        shutil.copy2(env_src, env_dest)
        log.info("Copied /data/.env → %s", env_dest)
elif not env_dest.exists():
    log.warning(".env not found at /data/.env or %s — API keys may be missing", env_dest)

# ── Step 3: Pre-download all 4 OSM models to /data/hf_cache ──────────────
PY = _venv_python()  # re-check in case step 2 somehow changed state
# Pure I/O — no GPU activity, no CUDA context.
# Eviction-safe: snapshot_download resumes from partial .incomplete blobs.
step("PREDOWNLOAD_OK", [PY, str(REPO / "akash" / "predownload_models.py")])

# ── Step 4: 2-seed GPU dry run ─────────────────────────────────────────────
# Loads all models from disk cache — zero network downloads during this phase.
# Sequential load/unload keeps peak VRAM ≈ 16 GB (one model at a time).
PY = _venv_python()
step(
    "DRYRUN_OK",
    [PY, str(REPO / "Code" / "mirage" / "Dry_Run" / "dry_run_gpu_cpu.py"),
     "--n-seeds", "2"],
    cwd=REPO / "Code" / "mirage",
)

# ── All done ──────────────────────────────────────────────────────────────
mark("PIPELINE_COMPLETE")
log.info("=== PIPELINE_COMPLETE — dry run passed. Ready for full production run. ===")
