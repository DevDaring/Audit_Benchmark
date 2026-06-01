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
import subprocess
import sys
import time

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

_REGEN_POLL_SEC = int(os.environ.get("MIRAGE_REGEN_POLL_SEC", "30"))
_REGEN_TIMEOUT_SEC = int(os.environ.get("MIRAGE_REGEN_TIMEOUT_SEC", "43200"))


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


def unmark(name: str) -> None:
    path = STATE / name
    if path.exists():
        path.unlink()
        log.warning("MARKER cleared: %s", name)


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


def _pentad_production_ready() -> bool:
    """Return True only when the full 12-slot pentad passes all gates."""
    if not PENTAD.exists():
        return False
    try:
        sys.path.insert(0, str(MIRAGE))
        import pandas as pd
        from Dataset.validate_pentad import assert_production_ready

        df = pd.read_parquet(PENTAD)
        assert_production_ready(df)
        log.info("Pentad production-ready: %d rows.", len(df))
        return True
    except Exception as exc:
        log.warning("Pentad NOT production-ready: %s", exc)
        return False


def _regen_in_progress() -> bool:
    r = subprocess.run(
        ["pgrep", "-f", "regenerate_api_slots.py"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _wait_for_regen() -> bool:
    """Block until regenerate_api_slots finishes or timeout."""
    deadline = time.time() + _REGEN_TIMEOUT_SEC
    while time.time() < deadline:
        if _pentad_production_ready():
            return True
        if not _regen_in_progress():
            return _pentad_production_ready()
        log.info(
            "Waiting for regenerate_api_slots.py (poll %ds)...",
            _REGEN_POLL_SEC,
        )
        time.sleep(_REGEN_POLL_SEC)
    log.error("Timed out waiting for API slot regeneration.")
    return False


def _det_slots_valid() -> bool:
    """True when a/b/c are complete and pass validation (d/e may still be missing)."""
    if not PENTAD.exists():
        return False
    try:
        sys.path.insert(0, str(MIRAGE))
        import pandas as pd
        from Dataset.validate_pentad import run_all_validations

        df = pd.read_parquet(PENTAD)
        audit = df[
            df["seed_source"].astype(str).str.lower().isin(
                {"bbq", "crows_pairs", "stereoset"}
            )
        ]
        n_seeds = int(audit["seed_id"].nunique())
        n_det = int((df["slot"].isin(["a", "b", "c"])).sum())
        if n_seeds < 596 or n_det < n_seeds * 7:
            return False
        run_all_validations(df, require_api_slots=False)
        log.info("Deterministic pentad valid: %d seeds, %d det rows.", n_seeds, n_det)
        return True
    except Exception as exc:
        log.warning("Deterministic pentad NOT valid: %s", exc)
        return False


def _ensure_dataset() -> None:
    """Build or repair pentad until production-ready; never start GPU on partial data."""
    if done("DATASET_OK") and not _pentad_production_ready():
        log.error("DATASET_OK marker is stale — pentad no longer valid.")
        unmark("DATASET_OK")
        unmark("GPU_PIPELINE_OK")

    if done("DATASET_OK"):
        log.info("SKIP DATASET_OK (marker present and pentad valid)")
        return

    if _pentad_production_ready():
        mark("DATASET_OK")
        return

    PY = _venv_python()

    if _det_slots_valid():
        log.info(
            "Deterministic slots valid — skipping patch_det_slots (preserves d/e + checkpoints)."
        )
    else:
        log.info("Deterministic slots invalid — running patch_det_slots.py ...")
        r = subprocess.run(
            [PY, str(MIRAGE / "patch_det_slots.py")],
            cwd=str(MIRAGE),
            env=_base_env(),
        )
        if r.returncode != 0:
            log.error("patch_det_slots.py failed — cannot proceed to GPU.")
            sys.exit(r.returncode)

    if _pentad_production_ready():
        from Dataset.validate_pentad import write_pentad_manifest
        import pandas as pd

        write_pentad_manifest(pd.read_parquet(PENTAD))
        mark("DATASET_OK")
        return

    if _regen_in_progress():
        log.info("regenerate_api_slots.py already running — waiting ...")
    else:
        ckpt_ctx = MIRAGE / "Dataset" / "seeds" / "context_shift_checkpoint.json"
        ckpt_cot = MIRAGE / "Dataset" / "seeds" / "cot_attack_checkpoint.json"
        regen_argv = [PY, str(MIRAGE / "regenerate_api_slots.py")]
        if ckpt_ctx.exists() or ckpt_cot.exists():
            regen_argv.append("--keep-checkpoint")
            log.info("Resuming regenerate_api_slots.py from checkpoint ...")
        else:
            log.info("Starting regenerate_api_slots.py ...")
        r = subprocess.run(
            regen_argv,
            cwd=str(MIRAGE),
            env=_base_env({"PYTHONUNBUFFERED": "1"}),
        )
        if r.returncode != 0 and not _regen_in_progress():
            log.error("regenerate_api_slots.py failed.")
            sys.exit(r.returncode)

    if not _wait_for_regen():
        log.error("Dataset build incomplete after regen wait.")
        sys.exit(1)

    from Dataset.validate_pentad import write_pentad_manifest
    import pandas as pd

    df = pd.read_parquet(PENTAD)
    from Dataset.validate_pentad import assert_production_ready

    assert_production_ready(df)
    write_pentad_manifest(df)
    mark("DATASET_OK")
    log.info("DATASET_OK — full pentad validated.")


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

# ── Step 4: Dataset build (CPU + DeepSeek API — NOT GPU) ──────────────────
_ensure_dataset()

# ── Step 5: Full GPU pipeline (GPU_CPU/ only) ─────────────────────────────
if done("GPU_PIPELINE_OK") and not _pentad_production_ready():
    unmark("GPU_PIPELINE_OK")

PY = _venv_python()
step(
    "GPU_PIPELINE_OK",
    [PY, str(MIRAGE / "GPU_CPU" / "run_gpu_pipeline.py")],
    cwd=MIRAGE,
)

mark("PIPELINE_COMPLETE")
log.info("=== PIPELINE_COMPLETE — production GPU pipeline finished ===")
