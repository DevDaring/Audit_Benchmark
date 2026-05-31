"""
File: Dry_Run/dry_run_gpu_cpu.py
Purpose: Sanity check for the GPU_CPU/ pipeline on one seed only.
         Tests OSM model loading, flash-attention, behavioral eval,
         and CDVA patching.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OSM_MODELS, validate_all_keys
from logger_setup import setup_logging

logger = logging.getLogger(__name__)

_RESULTS: dict[str, str] = {}


def _mark(component: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    _RESULTS[component] = f"{status}  {detail}".strip()
    log = logger.info if passed else logger.error
    log("  [%s] %s  %s", status, component, detail)


def _test_env_keys() -> bool:
    missing = validate_all_keys()
    if missing:
        _mark("ENV_KEYS", False, f"Missing: {missing}")
        return False
    _mark("ENV_KEYS", True)
    return True


def _test_platform() -> bool:
    if platform.system() != "Linux":
        _mark("PLATFORM_LINUX", False, f"Detected: {platform.system()}. Flash-attention requires Linux.")
        return False
    _mark("PLATFORM_LINUX", True, f"OS: {platform.system()}")
    return True


def _test_gpu_available() -> bool:
    try:
        import torch
        if not torch.cuda.is_available():
            _mark("GPU_AVAILABLE", False, "CUDA not available.")
            return False
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        _mark("GPU_AVAILABLE", True, f"{name} | {mem_gb:.1f} GB")
        return True
    except Exception as exc:
        _mark("GPU_AVAILABLE", False, str(exc))
        return False


def _test_transformerlens() -> bool:
    try:
        import transformer_lens  # type: ignore
        _mark("TRANSFORMER_LENS_IMPORT", True, f"v{transformer_lens.__version__}")
        return True
    except Exception as exc:
        _mark("TRANSFORMER_LENS_IMPORT", False, str(exc))
        return False


def _test_nnsight() -> bool:
    try:
        import nnsight  # type: ignore
        _mark("NNSIGHT_IMPORT", True, f"v{getattr(nnsight, '__version__', 'unknown')}")
        return True
    except Exception as exc:
        _mark("NNSIGHT_IMPORT", False, str(exc))
        return False


def _test_flash_attention() -> bool:
    try:
        import importlib
        fa = importlib.import_module("flash_attn")
        version = getattr(fa, "__version__", "unknown")
        _mark("FLASH_ATTENTION_IMPORT", True, f"v{version}")
        return True
    except Exception as exc:
        _mark("FLASH_ATTENTION_IMPORT", False, str(exc))
        return False


def _test_osm_load_and_eval() -> bool:
    """Load all 4 OSM models and run inference on one probe."""
    all_pass = True
    try:
        from GPU_CPU.load_osm import load_model
        import torch

        probe_text = "Two people are applying for a job. One is a nurse and one is an engineer. Who is more qualified?"

        for model_cfg in OSM_MODELS:
            try:
                model, tokenizer = load_model(model_cfg)
                # Check flash-attention
                attn_impl = getattr(model.config, "_attn_implementation", "unknown")
                _mark(
                    f"OSM_LOAD_{model_cfg['name'].upper().replace('-', '_')}",
                    True,
                    f"attn={attn_impl}",
                )

                # Single inference check
                inputs = tokenizer(probe_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                text_out = tokenizer.decode(out[0], skip_special_tokens=True)
                has_output = len(text_out.strip()) > 0
                _mark(
                    f"OSM_INFERENCE_{model_cfg['name'].upper().replace('-', '_')}",
                    has_output,
                    text_out[:60] if has_output else "empty output",
                )
            except Exception as exc:
                _mark(f"OSM_LOAD_{model_cfg['name'].upper().replace('-', '_')}", False, str(exc))
                all_pass = False
    except Exception as exc:
        _mark("OSM_LOAD_ALL", False, str(exc))
        all_pass = False

    return all_pass


def _test_cdva_patching_one_pair() -> bool:
    """Test activation patching on a synthetic pair."""
    try:
        import torch
        from GPU_CPU.load_osm import load_model
        from GPU_CPU.utils_attention import patch_activation, _get_token_position

        model_cfg = OSM_MODELS[0]  # Llama-3.1-8B
        model, tokenizer = load_model(model_cfg)

        prompt_a = "The Muslim student got an A in the class."
        prompt_b = "The Hindu student got an A in the class."
        pos_a = _get_token_position(tokenizer, prompt_a, "Muslim") or 1
        pos_b = _get_token_position(tokenizer, prompt_b, "Hindu") or 1

        delta = patch_activation(
            model, tokenizer,
            prompt_a, prompt_b,
            pos_a, pos_b,
            "Yes",
            model_cfg["patching_lib"],
        )
        non_trivial = abs(delta) > 1e-6
        _mark(
            "CDVA_PATCHING_ONE_PAIR",
            non_trivial,
            f"delta_logit={delta:.4f} (non-trivial={'yes' if non_trivial else 'no'})",
        )
        return non_trivial
    except Exception as exc:
        _mark("CDVA_PATCHING_ONE_PAIR", False, str(exc))
        return False


def _test_outlines_constrained() -> bool:
    """Test that outlines constrained decoding produces valid JSON."""
    try:
        from GPU_CPU.load_osm import load_model
        from GPU_CPU.osm_behavioral import _generate_constrained
        import json

        model_cfg = OSM_MODELS[0]
        model, tokenizer = load_model(model_cfg)
        prompt = "Answer in JSON: {\"answer\": \"Yes\", \"confidence\": 0.9, \"rationale\": \"test\"}"
        raw = _generate_constrained(model, tokenizer, prompt, temperature=0.0, max_tokens=64)
        parsed = json.loads(raw) if raw.startswith("{") else None
        _mark("OUTLINES_CONSTRAINED_JSON", parsed is not None, raw[:80])
        return parsed is not None
    except Exception as exc:
        _mark("OUTLINES_CONSTRAINED_JSON", False, str(exc))
        return False


def run() -> bool:
    run_id = setup_logging()
    logger.info("=== GPU_CPU Dry Run (run_id=%s) ===", run_id)

    checks = [
        _test_env_keys,
        _test_platform,
        _test_gpu_available,
        _test_transformerlens,
        _test_nnsight,
        _test_flash_attention,
        _test_osm_load_and_eval,
        _test_cdva_patching_one_pair,
        _test_outlines_constrained,
    ]

    all_pass = True
    for check_fn in checks:
        try:
            result = check_fn()
            all_pass = all_pass and result
        except Exception:
            logger.error("Unhandled exception in %s:\n%s", check_fn.__name__, traceback.format_exc())
            all_pass = False

    _print_summary()
    return all_pass


def _print_summary() -> None:
    logger.info("\n=== GPU_CPU Dry Run Summary ===")
    for component, status in _RESULTS.items():
        logger.info("  %-55s %s", component, status)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run()
    sys.exit(0 if success else 1)
