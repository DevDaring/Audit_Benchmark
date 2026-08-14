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
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, RANDOM_SEED, RESULTS_DIR, SEEDS_DIR  # noqa: E402
from GPU_CPU.load_osm import load_model, unload_model  # noqa: E402
from TIST.language_support import is_supported, skip_reason  # noqa: E402
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
# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------
# Lease 5 hung inside the behavioural generation at 504 of 2,644 prompts and sat there for
# 53 minutes on a billing GPU. Nothing detected it: the process was alive, the container
# was alive, and the sync daemon kept committing because it rewrites its own status file
# every cycle, so commit age carried no information about progress.
#
# This thread watches the two signals that do mean progress, the total number of result
# records and the modification time of this process's log, and kills the process when
# neither moves for WATCHDOG_STALL_MIN minutes. The bootstrap's supervisor retries on a
# non-zero exit after pulling, and every task resumes from its JSONL, so a kill costs at
# most one stalled interval rather than the whole run.
#
# os._exit is deliberate. A hang inside a CUDA call does not respond to an exception in
# another thread, and sys.exit only raises in the calling thread.
WATCHDOG_STALL_MIN = int(os.environ.get("TIST_WATCHDOG_STALL_MIN", "25"))
WATCHDOG_EXIT_CODE = 9


def _result_record_count() -> int:
    total = 0
    for sub in ("e1", "e4"):
        d = OUT / sub
        if not d.exists():
            continue
        for f in d.glob("*.jsonl"):
            try:
                with f.open("rb") as fh:
                    total += sum(1 for _ in fh)
            except OSError:
                pass
    return total


def start_watchdog(log_path: Path | None = None) -> None:
    import threading

    def _watch() -> None:
        last_change = time.time()
        last_sig = (-1, -1.0)
        while True:
            time.sleep(60)
            try:
                recs = _result_record_count()
                mtime = log_path.stat().st_mtime if log_path and log_path.exists() else 0.0
            except Exception:  # noqa: BLE001
                continue
            sig = (recs, mtime)
            if sig != last_sig:
                last_sig, last_change = sig, time.time()
                continue
            idle_min = (time.time() - last_change) / 60.0
            if idle_min >= WATCHDOG_STALL_MIN:
                log.error(
                    "WATCHDOG: no new records and no log activity for %.1f minutes "
                    "(records=%d). Treating this as a hang and exiting %d so the "
                    "supervisor restarts; completed work resumes from its JSONL.",
                    idle_min, recs, WATCHDOG_EXIT_CODE,
                )
                sys.stderr.flush()
                os._exit(WATCHDOG_EXIT_CODE)

    t = threading.Thread(target=_watch, name="tist-watchdog", daemon=True)
    t.start()
    log.info("watchdog armed: exits after %d min with no records and no log activity",
             WATCHDOG_STALL_MIN)


def _done(path: Path, key_fields: tuple) -> set:
    """
    Units already computed successfully. Failed rows are deliberately NOT counted.

    A failed row records an absence, not a result, and treating it as complete makes the
    failure permanent. That is what happened to Bengali: every pair recorded
    "position not found" against a tokenisation bug, and because those rows were treated
    as done, fixing the bug would not have retried a single one of them.

    Only ok=true units are skipped, so any fix to a failure mode takes effect on the next
    run with no file surgery. Units that genuinely cannot be computed are re-attempted
    each run, which is cheap: they fail in the lookup before any GPU work happens.
    """
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("ok") is False:
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
        # The memo holds this seed's prompts only. Clearing per seed bounds memory at a
        # few hundred megabytes while keeping every within-seed reuse.
        patcher.clear_memo()

        bias = _bias_answer(group)
        if not bias:
            continue
        bias_ids = patcher.token_ids(bias)
        if not bias_ids:
            continue
        bias_id = bias_ids[0]

        # One shuffled-control donor per seed rather than one per pair. The control asks
        # whether an unrelated seed's protected-token activation moves the logit, and a
        # donor drawn per seed answers that just as well while making its residual cache
        # reusable across the seed's twenty pairs. Across 596 seeds the donor still
        # varies widely, so the control keeps its variety.
        donor_prompt = donor_pos = donor_seed = None
        for _ in range(5):
            d_idx = int(rng.integers(0, len(seeds)))
            if d_idx == si:
                continue
            d_seed, d_group = seeds[d_idx]
            d_c = d_group[d_group["slot"] == "c"]
            if not len(d_c):
                continue
            d_row = d_c.iloc[0]
            d_pos = patcher.position_of(str(d_row["prompt_text"]), str(d_row["swap_token"]))
            if d_pos is not None:
                donor_prompt, donor_pos, donor_seed = str(d_row["prompt_text"]), d_pos, d_seed
                break

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
                if donor_prompt is not None:
                    d_cache = patcher.cache(donor_prompt)      # memoised across the seed
                    lp = patcher.patched_logits(d_cache, donor_pos, pb, pos_b)
                    rec["placebo_shuffled_single"] = metric_single_logit(base_b, lp, bias_id)
                    rec["placebo_shuffled_kl"] = metric_kl(base_b, lp)
                    rec["placebo_shuffled_donor"] = donor_seed

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
    for u_i, (seed_id, group, a, b, bias) in enumerate(units):
        if u_i % 25 == 0:
            patcher.clear_memo()
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
        patcher.clear_memo()
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
    for u_i, (seed_id, group, a, b, bias) in enumerate(units):
        if u_i % 25 == 0:
            patcher.clear_memo()
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
# Language competence probe: measure before auditing
# ---------------------------------------------------------------------------
def task_e4_competence(model_cfg: dict, model, tokenizer, limit: int) -> None:
    """
    Can this model actually read Hindi / Bengali? Decide by measurement, not by label.

    Scored on slot-a items whose gold answer is a specific option rather than the
    "cannot be determined" escape, so a model that always picks the escape cannot pass.
    The verdict is written to results/tist/e4/competence.json and gates e4_cdva and
    e4_behav for this model.
    """
    import uuid

    from GPU_CPU.osm_behavioral import evaluate_osm_model
    from CPU_Only.scoring import _answers_match
    from TIST.competence_probe import evaluate, probe_items, record

    name = model_cfg["name"]
    for lang in ("hi", "bn"):
        src = SEEDS_DIR / f"pentad_{lang}.parquet"
        if not src.exists():
            continue
        items = probe_items(pd.read_parquet(src))
        if limit:
            items = items.head(max(limit, 2))
        if items.empty:
            continue

        log.info("competence probe %s / %s on %d determinate items", name, lang, len(items))
        try:
            df = evaluate_osm_model(model_cfg, model, tokenizer, items,
                                    run_id=str(uuid.uuid4()), sample_index=0)
        except Exception as exc:  # noqa: BLE001
            log.error("competence probe failed for %s / %s: %s", name, lang, str(exc)[:200])
            continue

        n_ok = 0
        scorable = 0
        for _, r in df.iterrows():
            gold = str(r.get("gold_answer", ""))
            if not gold:
                continue
            scorable += 1
            if _answers_match(str(r.get("parsed_answer", "")), gold, str(r.get("seed_source", ""))):
                n_ok += 1

        verdict = evaluate(n_ok, scorable)
        verdict["lang"] = lang
        verdict["model_name"] = name
        record(RESULTS_DIR, name, lang, verdict)

# ---------------------------------------------------------------------------
# E4 multilingual CDVA
# ---------------------------------------------------------------------------
def task_e4_cdva(patcher: Patcher, model_name: str, limit: int) -> None:
    for lang in ("hi", "bn"):
        # A bias audit in a language the model was never built for measures language
        # competence, not bias. Skip rather than record a zero; see TIST/language_support.
        if not is_supported(model_name, lang, RESULTS_DIR):
            log.info("SKIP %s / %s: %s", model_name, lang, skip_reason(model_name, lang, RESULTS_DIR))
            continue
        src = SEEDS_DIR / f"pentad_{lang}.parquet"
        if not src.exists():
            log.warning("%s absent; skipping %s CDVA", src.name, lang)
            continue
        pen = pd.read_parquet(src)
        path = OUT / "e4" / f"cdva_{lang}_{model_name}.jsonl"
        done = _done(path, ("seed_id", "sub_a", "sub_b"))
        n = n_located = n_total = 0

        for seed_id, group in pen.groupby("seed_id"):
            patcher.clear_memo()
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
def _ensure_python_headers() -> None:
    """
    Make sure Python.h exists, so Triton can JIT-compile its CUDA helper.

    Without python3-dev, every batched generation call fails to compile and falls back to
    single-prompt decoding, several times slower over the behavioural pass. The container
    image ships without headers, so this installs them on first need. It is a no-op
    anywhere the headers are already present, and a failure here is not fatal: the
    evaluator still works, just slower.
    """
    import subprocess
    import sysconfig

    inc = sysconfig.get_paths().get("include", "")
    if inc and (Path(inc) / "Python.h").exists():
        return
    log.warning("Python.h missing; installing python3-dev so Triton can batch")
    try:
        subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "python3-dev"],
                       check=True, capture_output=True, timeout=600)
        log.info("python3-dev installed; batched generation available")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not install python3-dev (%s); generation stays single-prompt",
                    str(exc)[:160])


def task_e4_behav(model_cfg: dict, model, tokenizer, limit: int) -> None:
    """Reuses the production behavioural evaluator so scoring stays comparable."""
    import uuid

    _ensure_python_headers()

    from GPU_CPU.osm_behavioral import evaluate_osm_model

    for lang in ("hi", "bn"):
        if not is_supported(model_cfg["name"], lang, RESULTS_DIR):
            log.info("SKIP %s / %s behavioural: %s",
                     model_cfg["name"], lang, skip_reason(model_cfg["name"], lang, RESULTS_DIR))
            continue
        src = SEEDS_DIR / f"pentad_{lang}.parquet"
        if not src.exists():
            log.warning("%s absent; skipping %s behavioural", src.name, lang)
            continue
        pen = pd.read_parquet(src)
        expected = len(pen)
        if limit:
            pen = pen.head(limit)
        out_path = OUT / "e4" / f"behav_{lang}_{model_cfg['name']}.parquet"

        # Resume on completeness, not on existence. A dry run writes a two-row parquet to
        # this exact path, and those files reached the repository, so a fresh clone that
        # skipped on existence alone would present two rows as the full multilingual
        # behavioural result. Anything short of the expected prompt count is recomputed.
        if out_path.exists():
            try:
                have = len(pd.read_parquet(out_path))
            except Exception:  # noqa: BLE001
                have = -1
            if not limit and have < expected:
                log.warning(
                    "%s holds %d rows, expected %d; recomputing rather than trusting it",
                    out_path.name, have, expected,
                )
                out_path.unlink()
            else:
                log.info("%s complete (%d rows), skipping", out_path.name, have)
                continue
        log.info("%s behavioural %s: %d prompts", model_cfg["name"], lang, len(pen))
        df = evaluate_osm_model(
            model_cfg, model, tokenizer, pen, run_id=str(uuid.uuid4()), sample_index=0
        )
        df["lang"] = lang
        df.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
def _dispatch_parallel(tasks: list[str], models: list[dict], pool: int, dry: bool,
                       stagger: int) -> int:
    """
    Run one subprocess per model, at most `pool` at a time.

    Safe by construction. Each model is an independent computation: same weights, same
    prompts, same batch size of one, same kernels. Concurrent processes share only VRAM,
    and every task already writes to a per-model JSONL keyed by model name, so there is
    no shared output state. Results are identical to a sequential run.

    Batching several prompts into one forward pass is the other obvious speedup and is
    deliberately not done: padding changes reduction order and can select different
    kernels, which moves logits in the last bits. E1.1 turns on separating small effects
    from smaller ones, so a perturbed forward pass is not worth the wall-clock.

    VRAM on an 80 GB A100, bfloat16: llama ~32 GB (TransformerLens keeps a converted
    model beside the HF weights), qwen ~16 GB, gemma ~10 GB, phi ~9 GB. Largest first,
    so a heavy model pairs with a light one. Starts are staggered so two TransformerLens
    conversions never overlap.
    """
    import subprocess

    # Longest job first. Total wall time per model is dominated by the behavioural pass,
    # and Phi-4-mini decodes at 4.54 s/prompt against Llama's 0.99 in the production run,
    # which makes it about 12 h against 4.8 h even though it is the smallest model.
    # Ordering by VRAM put Phi last, so it started six hours in and set the finish time.
    # Longest-first also happens to be VRAM-friendly here: phi 9 GB + qwen 16 GB +
    # llama 32 GB is 57 GB of the 80 GB card, with gemma's 10 GB following on.
    order = {"phi-4-mini-instruct": 0, "qwen2.5-7b-instruct": 1,
             "gemma-2-2b-it": 2, "llama-3.1-8b-instruct": 3}
    names = sorted((m["name"] for m in models), key=lambda n: order.get(n, 99))

    logs = Path(__file__).resolve().parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).resolve()), "--tasks", *tasks, "--parallel", "1"]
    if dry:
        base.append("--dry")

    running: list[tuple] = []
    failures: list[str] = []
    queue = list(names)

    while queue or running:
        while queue and len(running) < pool:
            name = queue.pop(0)
            fh = (logs / f"run_{name}.log").open("w", encoding="utf-8")
            log.info("launching %s (%d running)", name, len(running) + 1)
            p = subprocess.Popen(base + ["--models", name], stdout=fh, stderr=subprocess.STDOUT)
            running.append((p, name, fh))
            if queue:
                time.sleep(stagger)

        time.sleep(10)
        for entry in list(running):
            p, name, fh = entry
            if p.poll() is None:
                continue
            fh.close()
            running.remove(entry)
            if p.returncode != 0:
                log.error("%s failed rc=%d (TIST/logs/run_%s.log)", name, p.returncode, name)
                failures.append(name)
            else:
                log.info("%s finished", name)

    if failures:
        log.warning("%d of %d models failed: %s", len(failures), len(names), ", ".join(failures))
    # Non-zero only if everything failed; a partial run still carries usable results.
    return 4 if len(failures) == len(names) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["all"])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dry", action="store_true", help="2 units per task on the real code path")
    ap.add_argument("--parallel", type=int, default=None,
                    help="models to run concurrently; default 2 when several are selected")
    ap.add_argument("--stagger", type=int, default=120,
                    help="seconds between process launches, so two TL conversions do not overlap")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    (OUT / "e1").mkdir(parents=True, exist_ok=True)
    (OUT / "e4").mkdir(parents=True, exist_ok=True)

    all_tasks = ["e1_battery", "e1_controls", "e1_layers", "e1_mediation",
                 "e4_competence", "e4_cdva", "e4_behav"]
    tasks = all_tasks if "all" in args.tasks else args.tasks
    models = [m for m in OSM_MODELS if not args.models or m["name"] in args.models]
    limit = 2 if args.dry else 0

    # Fan out over models unless this process is already a single-model worker.
    # Sequential by default. Multi-process fan-out does not survive on this provider.
    #
    # Evidence. Leases 1 and 2 ran all four models sequentially in one process and the dry
    # run completed normally. Every fan-out attempt died with SIGTERM and no Python
    # traceback shortly after a second process attached to the GPU: pool of 3 twice, then
    # pool of 2 with phi and qwen, which together hold about 25 GB of an 80 GB card and
    # load through NNsight without a TransformerLens conversion. Memory was the first
    # explanation and it does not fit that pairing. What fits is the provider allocating
    # the GPU as one indivisible unit and killing a second process that attaches to it.
    #
    # The orphaned children completed their work each time, which is why the parent, not
    # the workers, is the thing being killed.
    #
    # Per-seed memoisation still stands and is worth about 45% of the forward passes, so
    # sequential here is not the naive path. Set --parallel explicitly to override, but
    # only on a provider known to permit GPU sharing.
    pool = args.parallel if args.parallel is not None else 1
    if pool > 1 and len(models) > 1:
        log.info("dispatching %d models, %d concurrent", len(models), pool)
        sys.exit(_dispatch_parallel(tasks, models, pool, args.dry, args.stagger))

    # Arm before any model is loaded, so a hang during loading is caught too.
    _logs = Path(__file__).resolve().parent / "logs"
    _lp = _logs / (f"run_{models[0]['name']}.log" if len(models) == 1 else "main.log")
    start_watchdog(_lp)

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

        # Patching tasks run before the behavioural one. E1 is the evidence Reviewer 1
        # asked for, so it must not be blocked by a failure in the multilingual
        # generation pass. Each task is isolated: one failing task costs its own
        # results, not the rest of the model's work.
        def _run(label: str, fn, *fargs) -> None:
            if label not in tasks:
                return
            try:
                fn(*fargs)
            except Exception as exc:  # noqa: BLE001
                log.exception("task %s failed for %s: %s", label, name, str(exc)[:200])
                failures.append(f"{name}/{label}: {str(exc)[:160]}")

        try:
            needs_patcher = [t for t in tasks if t != "e4_behav"]
            if needs_patcher:
                patcher = Patcher(model, tokenizer, cfg["patching_lib"])
                log.info("%s: %d layers via %s", name, patcher.n_layers, patcher.lib)
                _run("e1_battery", task_e1_battery, patcher, name, pentad, limit)
                _run("e1_controls", task_e1_controls, patcher, name, limit)
                _run("e1_layers", task_e1_layers, patcher, name, pentad, limit)
                _run("e1_mediation", task_e1_mediation, patcher, name, pentad, limit)
                _run("e4_competence", task_e4_competence, cfg, model, tokenizer, limit)
                _run("e4_cdva", task_e4_cdva, patcher, name, limit)

            _run("e4_behav", task_e4_behav, cfg, model, tokenizer, limit)
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

    # A model that loaded but produced no valid rows is a failed run, not a partial one.
    #
    # The previous version exited 0 whenever any model succeeded. Qwen and Phi wrote 9,566
    # rows each in which EVERY row was an error, the whole NNsight path being broken, and
    # because Llama and Gemma were fine the run reported COMPLETE, the lease auto-closed,
    # and the record count climbed to 103,144 looking like progress. Half the study was
    # missing and nothing said so.
    empty_models = []
    for cfg in models:
        name = cfg["name"]
        n_ok = 0
        for sub in ("e1", "e4"):
            for f in (OUT / sub).glob(f"*{name}*.jsonl"):
                try:
                    with f.open(encoding="utf-8") as fh:
                        for line in fh:
                            if '"ok": true' in line or '"ok":true' in line:
                                n_ok += 1
                                break
                except OSError:
                    pass
                if n_ok:
                    break
            if n_ok:
                break
        if n_ok == 0:
            empty_models.append(name)

    if empty_models:
        log.error(
            "MODELS WITH NO VALID ROWS: %s. Every unit failed for these, so the run is "
            "incomplete regardless of the other models. Exiting 5 rather than reporting "
            "success.", ", ".join(empty_models),
        )
        sys.exit(5)
    if failures:
        log.warning("%d of %d models failed to load", len(failures), len(models))
        for f in failures:
            log.warning("  %s", f)

    log.info("all requested tasks complete (%d of %d models)", n_loaded, len(models))


if __name__ == "__main__":
    main()
