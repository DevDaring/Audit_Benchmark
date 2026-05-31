"""
File: Dataset/sample_seeds.py
Purpose: Stratified seed selection from the four source benchmarks with
         fixed RNG seed for full reproducibility.

Implements / builds on / cites:
  - Parrish et al. (2022). BBQ. Findings of ACL 2022.
  - Nangia et al. (2020). CrowS-Pairs. EMNLP 2020.
  - Nadeem et al. (2021). StereoSet. ACL-IJCNLP 2021.
  - Zhao et al. (2018). WinoBias. NAACL 2018.

RNG: numpy.random.default_rng(seed=20260101) -- fixed for reproducibility.
Dev set (50 seeds) is sampled separately, disjoint from the 870 main seeds.

Part of the MIRAGE codebase. See README.md for full project context.
"""

import hashlib
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_SEED, SEEDS_DIR, ensure_dirs

logger = logging.getLogger(__name__)

SEED_COUNTS = {
    "bbq": 270,         # 30 per category x 9
    "crows_pairs": 200,
    "stereoset": 200,
    "winobias": 200,    # held out
}
DEV_SEED_COUNT = 50  # disjoint dev set for tau calibration

_SEEDS_PATH = SEEDS_DIR / "seeds.parquet"
_MANIFEST_PATH = SEEDS_DIR / "seeds_manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sample_bbq(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample 30 seeds per BBQ category (270 total)."""
    per_cat = 30
    frames: list[pd.DataFrame] = []
    for cat in sorted(df["bbq_category"].unique()):
        cat_df = df[df["bbq_category"] == cat]
        n = min(per_cat, len(cat_df))
        sample = cat_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False)
        sample = sample.copy()
        sample["seed_source"] = "bbq"
        sample["seed_category"] = cat
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def _sample_crows(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample ~22 seeds per bias type (200 total)."""
    per_type = 22
    frames: list[pd.DataFrame] = []
    bias_col = "bias_type" if "bias_type" in df.columns else df.columns[0]
    for bt in sorted(df[bias_col].unique()):
        bt_df = df[df[bias_col] == bt]
        n = min(per_type, len(bt_df))
        sample = bt_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False)
        sample = sample.copy()
        sample["seed_source"] = "crows_pairs"
        sample["seed_category"] = bt
        frames.append(sample)
    combined = pd.concat(frames, ignore_index=True)
    # Pad/trim to exactly 200
    if len(combined) > 200:
        combined = combined.sample(n=200, random_state=int(rng.integers(0, 2**31)), replace=False)
    return combined.reset_index(drop=True)


def _sample_stereoset(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample balanced across 4 domains (200 total)."""
    per_domain = 50
    frames: list[pd.DataFrame] = []
    domain_col = "bias_type" if "bias_type" in df.columns else df.columns[0]
    for domain in sorted(df[domain_col].unique()):
        dom_df = df[df[domain_col] == domain]
        n = min(per_domain, len(dom_df))
        sample = dom_df.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False)
        sample = sample.copy()
        sample["seed_source"] = "stereoset"
        sample["seed_category"] = domain
        frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def _sample_winobias(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample 200 rows balanced type1/type2 and pro/anti."""
    per_group = 50
    frames: list[pd.DataFrame] = []
    for wtype in ["type1", "type2"]:
        for direction in ["pro", "anti"]:
            sub = df[(df["wino_type"] == wtype) & (df["stereo_direction"] == direction)]
            n = min(per_group, len(sub))
            sample = sub.sample(n=n, random_state=int(rng.integers(0, 2**31)), replace=False)
            sample = sample.copy()
            sample["seed_source"] = "winobias"
            sample["seed_category"] = "Gender"
            sample["seed_subcategory"] = f"{wtype}_{direction}"
            frames.append(sample)
    return pd.concat(frames, ignore_index=True)


def _assign_seed_ids(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df["seed_id"] = [f"{prefix}_{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
    return df


def sample_seeds(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the main seed set (870 seeds) and dev set (50 seeds).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (main_seeds, dev_seeds)
    """
    ensure_dirs()

    if _SEEDS_PATH.exists() and not force:
        logger.info("Seeds cache hit: %s", _SEEDS_PATH)
        main_seeds = pd.read_parquet(_SEEDS_PATH)
        dev_path = SEEDS_DIR / "dev_seeds.parquet"
        dev_seeds = pd.read_parquet(dev_path) if dev_path.exists() else pd.DataFrame()
        logger.info(
            "Loaded %d main seeds and %d dev seeds from cache.",
            len(main_seeds),
            len(dev_seeds),
        )
        return main_seeds, dev_seeds

    logger.info("Sampling seeds with RNG seed %d.", RANDOM_SEED)
    rng = np.random.default_rng(seed=RANDOM_SEED)

    from Dataset.download_bbq import download_bbq
    from Dataset.download_crows_pairs import download_crows_pairs
    from Dataset.download_stereoset import download_stereoset
    from Dataset.download_winobias import download_winobias

    bbq_df = download_bbq()
    crows_df = download_crows_pairs()
    stereo_df = download_stereoset()
    wino_df = download_winobias()

    bbq_seeds = _assign_seed_ids(_sample_bbq(bbq_df, rng), "bbq")
    crows_seeds = _assign_seed_ids(_sample_crows(crows_df, rng), "crows")
    stereo_seeds = _assign_seed_ids(_sample_stereoset(stereo_df, rng), "stereo")
    wino_seeds = _assign_seed_ids(_sample_winobias(wino_df, rng), "wino")

    main_seeds = pd.concat(
        [bbq_seeds, crows_seeds, stereo_seeds, wino_seeds], ignore_index=True
    )

    # Dev set: 50 disjoint seeds drawn from the non-WinoBias pool
    non_wino = main_seeds[main_seeds["seed_source"] != "winobias"]
    dev_seeds = non_wino.sample(n=min(DEV_SEED_COUNT, len(non_wino)), random_state=int(rng.integers(0, 2**31)), replace=False)
    main_seeds = main_seeds[~main_seeds["seed_id"].isin(dev_seeds["seed_id"])].reset_index(drop=True)

    # Integrity checks
    assert main_seeds["seed_id"].is_unique, "Duplicate seed_id detected."
    assert len(main_seeds) > 0, "No seeds produced."

    # Save
    main_seeds.to_parquet(_SEEDS_PATH, index=False)
    dev_seeds.to_parquet(SEEDS_DIR / "dev_seeds.parquet", index=False)

    # Store SHA-256 manifest
    sha = _sha256(_SEEDS_PATH)
    import json
    manifest = {"seeds_sha256": sha, "n_main": len(main_seeds), "n_dev": len(dev_seeds)}
    with open(_MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "Sampled %d main seeds + %d dev seeds. SHA-256: %s",
        len(main_seeds),
        len(dev_seeds),
        sha,
    )
    return main_seeds, dev_seeds


def verify_seeds_integrity() -> None:
    """Re-check SHA-256 of seeds file against stored manifest. Fail loudly if changed."""
    import json

    if not _SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds file not found: {_SEEDS_PATH}")
    if not _MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {_MANIFEST_PATH}")

    with open(_MANIFEST_PATH) as fh:
        manifest = json.load(fh)

    current_sha = _sha256(_SEEDS_PATH)
    if current_sha != manifest["seeds_sha256"]:
        raise RuntimeError(
            f"Seeds file integrity check FAILED. "
            f"Expected SHA-256: {manifest['seeds_sha256']}, "
            f"Got: {current_sha}"
        )
    logger.info("Seeds integrity check passed. SHA-256: %s", current_sha)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_s, dev_s = sample_seeds()
    logger.info("Main seeds: %d, Dev seeds: %d", len(main_s), len(dev_s))
