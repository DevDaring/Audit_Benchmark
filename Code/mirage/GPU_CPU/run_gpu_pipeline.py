"""
File: GPU_CPU/run_gpu_pipeline.py
Purpose: Production entry-point for the full MIRAGE GPU pipeline.
         Runs after the dataset has been built (Dataset/seeds/pentad_dataset.parquet exists).

Steps:
  1. Load all 4 OSM models simultaneously (~42 GB; A100 80 GB handles this).
  2. run_osm_behavioral  — behavioral evaluation on the full pentad dataset.
  3. run_cdva_patching   — causal activation patching on counterfactual (c) variants.
  4. Unload all models to free VRAM for any subsequent CPU post-processing.
  5. run_cdva_calibration — tau threshold calibration on the dev set.

Both behavioral and CDVA functions include incremental-save / resume logic: if the
process is killed mid-run (e.g. an eviction), re-running this script will skip
already-completed rows and continue from the last checkpoint.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the MIRAGE codebase. See README.md for full project context.
"""
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RESULTS_DIR, SEEDS_DIR, ensure_dirs
from logger_setup import setup_logging


def main() -> bool:
    run_id = setup_logging()
    logger.info("=== MIRAGE GPU Pipeline (run_id=%s) ===", run_id)
    ensure_dirs()
    t0 = time.monotonic()

    # ── 0. Verify pentad dataset exists ──────────────────────────────────
    pentad_path = SEEDS_DIR / "pentad_dataset.parquet"
    if not pentad_path.exists():
        logger.error(
            "Pentad dataset not found at %s. "
            "Run 'python run_dataset.py' first to build the dataset.",
            pentad_path,
        )
        return False

    import pandas as pd

    pentad_df = pd.read_parquet(pentad_path)

    # Hard gate — refuse GPU work on partial or invalid pentad.
    from Dataset.validate_pentad import assert_production_ready, write_pentad_manifest

    assert_production_ready(pentad_df)
    write_pentad_manifest(pentad_df)

    from GPU_CPU.pipeline_guards import clear_stale_gpu_results_if_pentad_changed

    state_dir = Path(os.environ.get("STATE_DIR", "/data/state"))
    clear_stale_gpu_results_if_pentad_changed(state_dir)

    logger.info(
        "Pentad dataset loaded: %d rows | %d unique seeds",
        len(pentad_df),
        pentad_df["seed_id"].nunique(),
    )

    # ── 1. Load all 4 OSM models simultaneously ────────────────────────────
    logger.info("Step 1/4: Loading all 4 OSM models (~42 GB on A100 80 GB) ...")
    from GPU_CPU.load_osm import load_all_osm_models, unload_model
    models = load_all_osm_models()
    logger.info(
        "Step 1/4 done: %d models loaded (%.1f s elapsed)",
        len(models), time.monotonic() - t0,
    )

    # ── 2. Behavioral evaluation ───────────────────────────────────────────
    logger.info(
        "Step 2/4: Behavioral evaluation — %d models × %d rows ...",
        len(models), len(pentad_df),
    )
    from GPU_CPU.osm_behavioral import run_osm_behavioral
    behavioral_df = run_osm_behavioral(pentad_df, models, run_id)
    logger.info(
        "Step 2/4 done: behavioral_results.parquet written (%d rows, %.1f s)",
        len(behavioral_df), time.monotonic() - t0,
    )

    # ── 3. CDVA patching ───────────────────────────────────────────────────
    logger.info("Step 3/4: CDVA activation patching ...")
    from GPU_CPU.cdva_patching import run_cdva
    cdva_df = run_cdva(pentad_df, models, run_id)
    logger.info(
        "Step 3/4 done: cdva_results.parquet written (%d rows, %.1f s)",
        len(cdva_df), time.monotonic() - t0,
    )

    # ── 4. Unload models ───────────────────────────────────────────────────
    logger.info("Step 4/4: Unloading all OSM models ...")
    for model_cfg in OSM_MODELS:
        unload_model(model_cfg["name"])
    import torch, gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("VRAM freed. Running CDVA tau calibration on CPU ...")

    # ── 5. CDVA tau calibration (uses dev-seed subset of full results) ─────
    # calibrate_tau() takes the behavioral and CDVA dataframes filtered to the
    # dev seed set.  If dev seeds are not in the main pentad (det-only build),
    # this step logs a warning and produces a fallback tau = 0.5.
    from GPU_CPU.cdva_calibration import calibrate_tau
    dev_seeds_path = SEEDS_DIR / "dev_seeds.parquet"
    if dev_seeds_path.exists() and len(behavioral_df) > 0 and len(cdva_df) > 0:
        dev_seeds_df = pd.read_parquet(dev_seeds_path)
        dev_ids = set(dev_seeds_df["seed_id"].tolist()) if "seed_id" in dev_seeds_df.columns else set()
        dev_beh  = behavioral_df[behavioral_df["seed_id"].isin(dev_ids)] if dev_ids else behavioral_df
        dev_cdva = cdva_df[cdva_df["seed_id"].isin(dev_ids)] if dev_ids else cdva_df
        if len(dev_beh) > 0:
            calibrate_tau(dev_beh, dev_cdva)
        else:
            logger.warning("No dev-seed rows in behavioral results; tau calibration skipped.")
    else:
        logger.warning("dev_seeds.parquet not found; skipping tau calibration.")

    elapsed = time.monotonic() - t0
    logger.info(
        "=== GPU PIPELINE COMPLETE in %.1f s (%.1f h) ===",
        elapsed, elapsed / 3600,
    )
    logger.info("  Results in: %s", RESULTS_DIR)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
