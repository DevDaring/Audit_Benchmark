"""
File: TIST/run_tist_gpu.py
Purpose: Single entry point for every GPU task of the TIST resubmission.

Tasks, in the order the mission file prioritises them:

  e1_battery   E1.1 placebo controls, E1.3 patch direction, E1.6 metric robustness.
               Fused into one pass: all three vary what is done with a pair, and the
               expensive part (caching the two clean runs) is shared. Full final-position
               logits are captured once, so the three metrics cost nothing extra.
  e1_layers    E1.2 layer-wise localisation sweep, windowed, on a stratified subsample.
  e1_controls  E1.4 behavioural pass and CDVA over the synthetic ground-truth controls.
  e1_mediation E1.5 causal mediation, natural indirect effect, gender subset.
  e4_cdva      Multilingual CDVA on the Hindi and Bengali pentads.
  e4_behav     Multilingual behavioural pentad on the open models.

Every task writes JSONL incrementally, one record per unit of work, and skips units
already present in the file. A lease that dies mid-task resumes without repeating work.

Usage:
  python TIST/run_tist_gpu.py --tasks all
  python TIST/run_tist_gpu.py --tasks e1_battery --models llama-3.1-8b-instruct
  python TIST/run_tist_gpu.py --tasks all --dry     # 2 units per task, on the real code path

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR, SEEDS_DIR  # noqa: E402
from GPU_CPU.load_osm import load_model, unload_model  # noqa: E402
from TIST.patch_core import (  # noqa: E402
    Patcher,
    choose_placebo_positions,
    metric_kl,
    metric_logit_diff,
    metric_single_logit,
)

log = logging.getLogger("tist_gpu")

OUT = RESULTS_DIR / "tist"
LAYER_WINDOW = 4
LAYER_STRIDE = 4
LAYER_SUBSAMPLE = 1200      # pairs per model for the sweep
MEDIATION_SUBSAMPLE = 600


# ---------------------------------------------------------------------------
# JSONL checkpointing
# ---------------------------------------------------------------------------
def _done(path: Path, key_fields: tuple) -> set:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(tuple(r.get(k) for k in key_fields))
    return keys


def _emit(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")


# ---------------------------------------------------------------------------
# Shared pair construction: identical to what the production CDVA scored
# ---------------------------------------------------------------------------
def _pentad(clean: bool = True) -> pd.DataFrame:
    path = SEEDS_DIR / ("pentad_dataset_clean.parquet" if clean else "pentad_dataset.parquet")
    if not path.exists():
        path = SEEDS_DIR / "pentad_dataset.parquet"
        log.warning("clean pentad absent, falling back to %s", path.name)
    return pd.read_parquet(path)


def _bias_answer(group: pd.DataFrame) -> str:
    """Mirrors GPU_CPU.cdva_patching._get_bias_answer."""
    gold = str(group["gold_answer"].dropna().iloc[0]).strip() if len(group) else ""
    if gold and gold.lower() != "unknown":
        return gold.split()[0] if gold.split() else gold
    toks = group["swap_token"].dropna().astype(str)
    return str(toks.iloc[0]) if len(toks) else ""


def _pairs_for_seed(group: pd.DataFrame) -> list[tuple]:
    """Ordered slot-c variant pairs (A, B) for one seed, as the production audit built them."""
    c = group[group["slot"] == "c"].drop_duplicates(subset=["subvariant"])
    variants = list(c.itertuples())
    out = []
    for i in range(len(variants)):
        for j in range(len(variants)):
            if i == j:
                continue
            a, b = variants[i], variants[j]
            if str(a.prompt_text) == str(b.prompt_text):
                continue
            out.append((a, b))
    return out


def _anti_option(prompt_text: str, gold: str) -> str:
    """The first option that is not the gold answer, used for the logit-difference metric."""
    for line in str(prompt_text).splitlines():
        if line.startswith(("(A)", "(B)", "(C)")):
            opt = line[3:].strip()
            if opt.lower() != str(gold).strip().lower():
                return opt
    return ""


# ---------------------------------------------------------------------------
# E1.1 + E1.3 + E1.6
# ---------------------------------------------------------------------------
def task_e1_battery(patcher: Patcher, model_name: str, pentad: pd.DataFrame, limit: int) -> None:
    path = OUT / "e1" / f"battery_{model_name}.jsonl"
    done = _done(path, ("seed_id", "sub_a", "sub_b"))
    rng = np.random.default_rng(RANDOM_SEED)

    # Shuffled-pair control needs a donor cache from an unrelated seed.
    seeds = list(pentad.groupby("seed_id"))
    n_units = 0
    t0 = time.time()

    for si, (seed_id, group) in enumerate(seeds):
        bias = _bias_answer(group)
        if not bias:
            continue
        bias_ids = patcher.token_ids(bias)
        if not bias_ids:
            continue
        bias_id = bias_ids[0]

        for a, b in _pairs_for_seed(group):
            if (seed_id, a.subvariant, b.subvariant) in done:
                continue
            if limit and n_units >= limit:
                return

            pa, pb = str(a.prompt_text), str(b.prompt_text)
            tok_a, tok_b = str(a.swap_token), str(b.swap_token)
            pos_a = patcher.position_of(pa, tok_a)
            pos_b = patcher.position_of(pb, tok_b)
            if pos_a is None or pos_b is None:
                _emit(path, {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "ok": False, "reason": "position not found",
                })
                continue

            try:
                cache_a = patcher.cache(pa)
                base_b = patcher.logits(pb)

                anti = _anti_option(pb, str(b.gold_answer))
                anti_ids = patcher.token_ids(anti) if anti else []
                anti_id = anti_ids[0] if anti_ids else None

                rec = {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "ok": True,
                    "seed_source": str(group["seed_source"].iloc[0]),
                    "seed_category": str(group["seed_category"].iloc[0]),
                    "bias_answer": bias, "pos_a": pos_a, "pos_b": pos_b,
                }

                # --- E1.1 condition 0: the real patch (denoising, a -> b) ----
                pl = patcher.patched_logits(cache_a, pos_a, pb, pos_b)
                rec["real_single"] = metric_single_logit(base_b, pl, bias_id)
                rec["real_kl"] = metric_kl(base_b, pl)
                if anti_id is not None:
                    rec["real_logitdiff"] = metric_logit_diff(base_b, pl, bias_id, anti_id)

                # --- E1.1 conditions 1 and 2: within-prompt placebo positions ----
                plac = choose_placebo_positions(patcher, pb, pos_b, rng)
                for label, tgt in (("content", plac["content_pos"]), ("function", plac["function_pos"])):
                    if tgt is None:
                        rec[f"placebo_{label}_single"] = None
                        continue
                    # Source from the same non-protected position in the donor run,
                    # so only the position identity differs from the real patch.
                    src = tgt if tgt < min(len(v) for v in cache_a.values()) else pos_a
                    lp = patcher.patched_logits(cache_a, src, pb, tgt)
                    rec[f"placebo_{label}_single"] = metric_single_logit(base_b, lp, bias_id)
                    rec[f"placebo_{label}_kl"] = metric_kl(base_b, lp)
                    rec[f"placebo_{label}_pos"] = tgt

                # --- E1.1 condition 3: shuffled pair, unrelated seed's protected token ----
                donor_idx = int(rng.integers(0, len(seeds)))
                if donor_idx == si:
                    donor_idx = (donor_idx + 1) % len(seeds)
                d_seed, d_group = seeds[donor_idx]
                d_c = d_group[d_group["slot"] == "c"]
                if len(d_c):
                    d_row = d_c.iloc[0]
                    d_prompt, d_tok = str(d_row["prompt_text"]), str(d_row["swap_token"])
                    d_pos = patcher.position_of(d_prompt, d_tok)
                    if d_pos is not None:
                        d_cache = patcher.cache(d_prompt)
                        lp = patcher.patched_logits(d_cache, d_pos, pb, pos_b)
                        rec["placebo_shuffled_single"] = metric_single_logit(base_b, lp, bias_id)
                        rec["placebo_shuffled_kl"] = metric_kl(base_b, lp)
                        rec["placebo_shuffled_donor"] = d_seed

                # --- E1.3: the noising direction, b -> a ----
                cache_b = patcher.cache(pb)
                base_a = patcher.logits(pa)
                lp = patcher.patched_logits(cache_b, pos_b, pa, pos_a)
                rec["reverse_single"] = metric_single_logit(base_a, lp, bias_id)
                rec["reverse_kl"] = metric_kl(base_a, lp)

                _emit(path, rec)
            except Exception as exc:  # noqa: BLE001
                _emit(path, {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "ok": False, "reason": str(exc)[:200],
                })

            n_units += 1
            if n_units % 200 == 0:
                rate = n_units / max(time.time() - t0, 1e-6)
                log.info("%s battery: %d units, %.2f units/s", model_name, n_units, rate)


# ---------------------------------------------------------------------------
# E1.2 layer sweep
# ---------------------------------------------------------------------------
def task_e1_layers(patcher: Patcher, model_name: str, pentad: pd.DataFrame, limit: int) -> None:
    path = OUT / "e1" / f"layersweep_{model_name}.jsonl"
    done = _done(path, ("seed_id", "sub_a", "sub_b", "window_start"))
    rng = np.random.default_rng(RANDOM_SEED)

    units = []
    for seed_id, group in pentad.groupby("seed_id"):
        bias = _bias_answer(group)
        if not bias:
            continue
        for a, b in _pairs_for_seed(group):
            units.append((seed_id, group, a, b, bias))
    if len(units) > LAYER_SUBSAMPLE:
        idx = rng.choice(len(units), size=LAYER_SUBSAMPLE, replace=False)
        units = [units[i] for i in sorted(idx)]

    windows = list(range(0, patcher.n_layers, LAYER_STRIDE))
    n = 0
    for seed_id, group, a, b, bias in units:
        pa, pb = str(a.prompt_text), str(b.prompt_text)
        pos_a = patcher.position_of(pa, str(a.swap_token))
        pos_b = patcher.position_of(pb, str(b.swap_token))
        if pos_a is None or pos_b is None:
            continue
        ids = patcher.token_ids(bias)
        if not ids:
            continue
        bias_id = ids[0]

        try:
            cache_a = patcher.cache(pa)
            base_b = patcher.logits(pb)
        except Exception as exc:  # noqa: BLE001
            log.warning("layer sweep cache failed on %s: %s", seed_id, str(exc)[:120])
            continue

        for w in windows:
            if (seed_id, a.subvariant, b.subvariant, w) in done:
                continue
            if limit and n >= limit:
                return
            layers = list(range(w, min(w + LAYER_WINDOW, patcher.n_layers)))
            try:
                lp = patcher.patched_logits(cache_a, pos_a, pb, pos_b, layers=layers)
                _emit(path, {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "window_start": w,
                    "window_layers": layers, "n_layers": patcher.n_layers,
                    "delta_single": metric_single_logit(base_b, lp, bias_id),
                    "delta_kl": metric_kl(base_b, lp), "ok": True,
                })
            except Exception as exc:  # noqa: BLE001
                _emit(path, {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "window_start": w, "ok": False,
                    "reason": str(exc)[:200],
                })
            n += 1
        if n and n % 400 == 0:
            log.info("%s layer sweep: %d window-units", model_name, n)


# ---------------------------------------------------------------------------
# E1.4 synthetic controls
# ---------------------------------------------------------------------------
def task_e1_controls(patcher: Patcher, model_name: str, limit: int) -> None:
    path = OUT / "e1" / f"controls_{model_name}.jsonl"
    src = SEEDS_DIR / "synthetic_controls.parquet"
    if not src.exists():
        log.error("controls absent; run TIST/e1_4_build_controls.py first")
        return
    ctrl = pd.read_parquet(src)
    done = _done(path, ("seed_id", "sub_a", "sub_b"))
    n = 0

    for seed_id, group in ctrl.groupby("seed_id"):
        bias = _bias_answer(group)
        ids = patcher.token_ids(bias)
        if not ids:
            continue
        bias_id = ids[0]
        ctype = str(group["control_type"].iloc[0])

        for a, b in _pairs_for_seed(group):
            if (seed_id, a.subvariant, b.subvariant) in done:
                continue
            if limit and n >= limit:
                return
            pa, pb = str(a.prompt_text), str(b.prompt_text)
            pos_a = patcher.position_of(pa, str(a.swap_token))
            pos_b = patcher.position_of(pb, str(b.swap_token))
            if pos_a is None or pos_b is None:
                _emit(path, {"seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                             "model_name": model_name, "control_type": ctype,
                             "ok": False, "reason": "position not found"})
                n += 1
                continue
            try:
                cache_a = patcher.cache(pa)
                base_b = patcher.logits(pb)
                lp = patcher.patched_logits(cache_a, pos_a, pb, pos_b)
                _emit(path, {
                    "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                    "model_name": model_name, "control_type": ctype,
                    "seed_category": str(group["seed_category"].iloc[0]),
                    "bias_answer": bias, "ok": True,
                    "delta_single": metric_single_logit(base_b, lp, bias_id),
                    "delta_kl": metric_kl(base_b, lp),
                })
            except Exception as exc:  # noqa: BLE001
                _emit(path, {"seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                             "model_name": model_name, "control_type": ctype,
                             "ok": False, "reason": str(exc)[:200]})
            n += 1
    log.info("%s controls: %d units", model_name, n)


# ---------------------------------------------------------------------------
# E1.5 causal mediation (Vig et al.)
# ---------------------------------------------------------------------------
def task_e1_mediation(patcher: Patcher, model_name: str, pentad: pd.DataFrame, limit: int) -> None:
    """
    Natural indirect effect through the protected-token representation.

    Total effect  : logit(bias | prompt_b) - logit(bias | prompt_a)
    Indirect      : logit(bias | prompt_a with b's protected representation) - logit(bias | prompt_a)
    Direct        : total - indirect

    The commutator this audit reports is a patching quantity; the NIE is the mediation
    quantity Vig et al. define. Correlating them gives convergent evidence from an
    independent published method.
    """
    path = OUT / "e1" / f"mediation_{model_name}.jsonl"
    done = _done(path, ("seed_id", "sub_a", "sub_b"))
    rng = np.random.default_rng(RANDOM_SEED)

    gender = pentad[pentad["seed_category"].astype(str).str.contains("gender", case=False, na=False)]
    if gender.empty:
        log.warning("no gender-axis seeds found; mediation skipped")
        return

    units = []
    for seed_id, group in gender.groupby("seed_id"):
        bias = _bias_answer(group)
        if not bias:
            continue
        for a, b in _pairs_for_seed(group):
            units.append((seed_id, group, a, b, bias))
    if len(units) > MEDIATION_SUBSAMPLE:
        idx = rng.choice(len(units), size=MEDIATION_SUBSAMPLE, replace=False)
        units = [units[i] for i in sorted(idx)]

    n = 0
    for seed_id, group, a, b, bias in units:
        if (seed_id, a.subvariant, b.subvariant) in done:
            continue
        if limit and n >= limit:
            return
        pa, pb = str(a.prompt_text), str(b.prompt_text)
        pos_a = patcher.position_of(pa, str(a.swap_token))
        pos_b = patcher.position_of(pb, str(b.swap_token))
        ids = patcher.token_ids(bias)
        if pos_a is None or pos_b is None or not ids:
            n += 1
            continue
        bias_id = ids[0]
        try:
            base_a = patcher.logits(pa)
            base_b = patcher.logits(pb)
            cache_b = patcher.cache(pb)
            mediated = patcher.patched_logits(cache_b, pos_b, pa, pos_a)

            total = float(base_b[bias_id] - base_a[bias_id])
            indirect = float(mediated[bias_id] - base_a[bias_id])
            _emit(path, {
                "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                "model_name": model_name, "ok": True, "bias_answer": bias,
                "total_effect": total, "indirect_effect": indirect,
                "direct_effect": total - indirect,
            })
        except Exception as exc:  # noqa: BLE001
            _emit(path, {"seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                         "model_name": model_name, "ok": False, "reason": str(exc)[:200]})
        n += 1
    log.info("%s mediation: %d units", model_name, n)


# ---------------------------------------------------------------------------
# E4 multilingual CDVA
# ---------------------------------------------------------------------------
def task_e4_cdva(patcher: Patcher, model_name: str, limit: int) -> None:
    for lang in ("hi", "bn"):
        src = SEEDS_DIR / f"pentad_{lang}.parquet"
        if not src.exists():
            log.warning("%s absent; skipping %s CDVA", src.name, lang)
            continue
        pen = pd.read_parquet(src)
        path = OUT / "e4" / f"cdva_{lang}_{model_name}.jsonl"
        done = _done(path, ("seed_id", "sub_a", "sub_b"))
        n = n_located = n_total = 0

        for seed_id, group in pen.groupby("seed_id"):
            bias = _bias_answer(group)
            ids = patcher.token_ids(bias) if bias else []
            if not ids:
                continue
            bias_id = ids[0]
            for a, b in _pairs_for_seed(group):
                if (seed_id, a.subvariant, b.subvariant) in done:
                    continue
                if limit and n >= limit:
                    break
                n_total += 1
                pa, pb = str(a.prompt_text), str(b.prompt_text)
                pos_a = patcher.position_of(pa, str(a.swap_token))
                pos_b = patcher.position_of(pb, str(b.swap_token))
                if pos_a is None or pos_b is None:
                    # Expected for multi-token Devanagari and Bengali terms; the located
                    # fraction is itself a reported result, so failures are recorded.
                    _emit(path, {"seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                                 "model_name": model_name, "lang": lang, "ok": False,
                                 "reason": "position not found"})
                    n += 1
                    continue
                n_located += 1
                try:
                    cache_a = patcher.cache(pa)
                    base_b = patcher.logits(pb)
                    lp = patcher.patched_logits(cache_a, pos_a, pb, pos_b)
                    _emit(path, {
                        "seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                        "model_name": model_name, "lang": lang, "ok": True,
                        "seed_category": str(group["seed_category"].iloc[0]),
                        "bias_answer": bias,
                        "delta_single": metric_single_logit(base_b, lp, bias_id),
                        "delta_kl": metric_kl(base_b, lp),
                    })
                except Exception as exc:  # noqa: BLE001
                    _emit(path, {"seed_id": seed_id, "sub_a": a.subvariant, "sub_b": b.subvariant,
                                 "model_name": model_name, "lang": lang, "ok": False,
                                 "reason": str(exc)[:200]})
                n += 1
        log.info("%s %s CDVA: %d units, located %d/%d", model_name, lang, n, n_located, n_total)


# ---------------------------------------------------------------------------
# E4 multilingual behavioural
# ---------------------------------------------------------------------------
def task_e4_behav(model_cfg: dict, limit: int) -> None:
    """Reuses the production behavioural evaluator so scoring stays comparable."""
    from GPU_CPU.osm_behavioral import evaluate_osm_model

    for lang in ("hi", "bn"):
        src = SEEDS_DIR / f"pentad_{lang}.parquet"
        if not src.exists():
            log.warning("%s absent; skipping %s behavioural", src.name, lang)
            continue
        pen = pd.read_parquet(src)
        if limit:
            pen = pen.head(limit)
        out_path = OUT / "e4" / f"behav_{lang}_{model_cfg['name']}.parquet"
        if out_path.exists():
            log.info("%s exists, skipping", out_path.name)
            continue
        log.info("%s behavioural %s: %d prompts", model_cfg["name"], lang, len(pen))
        df = evaluate_osm_model(model_cfg, pen)
        df["lang"] = lang
        df.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dry", action="store_true", help="2 units per task on the real code path")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (OUT / "e1").mkdir(parents=True, exist_ok=True)
    (OUT / "e4").mkdir(parents=True, exist_ok=True)

    all_tasks = ["e1_battery", "e1_controls", "e1_layers", "e1_mediation", "e4_cdva", "e4_behav"]
    tasks = all_tasks if "all" in args.tasks else args.tasks
    models = [m for m in OSM_MODELS if not args.models or m["name"] in args.models]
    limit = 2 if args.dry else 0

    pentad = _pentad(clean=True)
    log.info("pentad: %d prompts, %d seeds", len(pentad), pentad["seed_id"].nunique())

    n_loaded = 0
    failures: list[str] = []

    for cfg in models:
        name = cfg["name"]
        log.info("=== %s ===", name)
        try:
            model, tokenizer = load_model(cfg)
            n_loaded += 1
        except Exception as exc:  # noqa: BLE001
            log.error("load failed for %s: %s", name, str(exc)[:300])
            failures.append(f"{name}: {str(exc)[:200]}")
            continue

        try:
            if "e4_behav" in tasks:
                task_e4_behav(cfg, limit)

            needs_patcher = [t for t in tasks if t != "e4_behav"]
            if needs_patcher:
                patcher = Patcher(model, tokenizer, cfg["patching_lib"])
                log.info("%s: %d layers via %s", name, patcher.n_layers, patcher.lib)
                if "e1_battery" in tasks:
                    task_e1_battery(patcher, name, pentad, limit)
                if "e1_controls" in tasks:
                    task_e1_controls(patcher, name, limit)
                if "e1_layers" in tasks:
                    task_e1_layers(patcher, name, pentad, limit)
                if "e1_mediation" in tasks:
                    task_e1_mediation(patcher, name, pentad, limit)
                if "e4_cdva" in tasks:
                    task_e4_cdva(patcher, name, limit)
        finally:
            unload_model(name)

    # A run that loaded no model did no work. Exiting zero there let the lease report
    # COMPLETE after producing nothing, which is worse than crashing: the supervisor
    # stopped retrying and the GPU sat billing. Fail loudly instead, so the bootstrap
    # loop pulls any fix and tries again.
    if n_loaded == 0:
        log.error("no model loaded; nothing was computed")
        for f in failures:
            log.error("  %s", f)
        sys.exit(3)
    if failures:
        log.warning("%d of %d models failed to load", len(failures), len(models))
        for f in failures:
            log.warning("  %s", f)

    log.info("all requested tasks complete (%d of %d models)", n_loaded, len(models))


if __name__ == "__main__":
    main()
