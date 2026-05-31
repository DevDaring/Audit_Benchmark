"""
File: GPU_CPU/load_osm.py
Purpose: Load all 4 OSM models in bf16 with flash-attention-2. Verifies
         flash-attention is active for each loaded model.

Implements / builds on / cites:
  - Kalaitzidis (2026). "The Evaluation Trap." arXiv:2605.14167
  - Dao et al. (2022). "FlashAttention." NeurIPS 2022.
  - MIRAGE OSM stack: Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9b, Phi-4-mini.

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HUGGINGFACE_TOKEN, OSM_MODELS

logger = logging.getLogger(__name__)

_LOADED_MODELS: dict[str, tuple[Any, Any]] = {}  # name -> (model, tokenizer)


def _check_platform() -> None:
    """Flash-attention requires Linux x86_64. Error out on other platforms."""
    system = platform.system()
    if system != "Linux":
        raise RuntimeError(
            f"Flash-attention-2 is only supported on Linux x86_64. "
            f"Detected OS: {system}. "
            "Run on Ubuntu 22.04/24.04 with an NVIDIA GPU (CUDA 12.4)."
        )


def _verify_flash_attention(model: Any, model_name: str) -> None:
    """Print which attention implementation is active."""
    try:
        cfg = model.config
        attn_impl = getattr(cfg, "_attn_implementation", "unknown")
        logger.info("  Model %-35s | attention impl: %s", model_name, attn_impl)
        if attn_impl != "flash_attention_2":
            logger.warning(
                "  WARNING: %s did not load with flash_attention_2 (got '%s').",
                model_name,
                attn_impl,
            )
    except Exception as exc:
        logger.warning("  Could not verify attention impl for %s: %s", model_name, exc)


def load_model(model_cfg: dict, force_reload: bool = False) -> tuple[Any, Any]:
    """
    Load a single OSM model and its tokenizer.

    Parameters
    ----------
    model_cfg : dict
        Entry from config.OSM_MODELS.
    force_reload : bool
        Re-load even if already in the in-process cache.

    Returns
    -------
    tuple[model, tokenizer]
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]

    if name in _LOADED_MODELS and not force_reload:
        logger.info("Model '%s' already loaded; returning cached instance.", name)
        return _LOADED_MODELS[name]

    logger.info("Loading model: %s (%s) ...", name, hf_id)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id,
        token=HUGGINGFACE_TOKEN,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _verify_flash_attention(model, name)
    _LOADED_MODELS[name] = (model, tokenizer)
    logger.info("Model '%s' loaded successfully.", name)
    return model, tokenizer


def unload_model(name: str) -> None:
    """
    Remove a model from the in-process cache and free its VRAM.

    Call this before loading a TransformerLens HookedTransformer on top of the
    same model weights to avoid an OOM (A100 40 GB is tight when both the HF
    model and the TL copy coexist for the 9 B Gemma model).
    """
    import gc
    import torch

    if name in _LOADED_MODELS:
        model, _ = _LOADED_MODELS.pop(name)
        try:
            del model
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model '%s' unloaded and VRAM freed.", name)
    else:
        logger.debug("unload_model: '%s' not in cache; nothing to do.", name)

    # Also clear the TransformerLens cache entry so a fresh TL model can be
    # created after a reload.
    try:
        from GPU_CPU.utils_attention import _TL_MODEL_CACHE
        keys_to_remove = [k for k in _TL_MODEL_CACHE if name.lower() in k.lower()]
        for k in keys_to_remove:
            del _TL_MODEL_CACHE[k]
    except Exception:
        pass


def load_all_osm_models() -> dict[str, tuple[Any, Any]]:
    """
    Load all 4 OSM models. Verifies GPU is available before starting.

    Returns
    -------
    dict[str, tuple[model, tokenizer]]
        Keys are model logical names.
    """
    _check_platform()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. OSM models require a NVIDIA GPU with CUDA 12.4."
        )

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info(
        "GPU: %s | Memory: %.1f GB",
        torch.cuda.get_device_name(0),
        gpu_mem_gb,
    )
    if gpu_mem_gb < 20:
        logger.warning(
            "GPU has only %.1f GB; some models may OOM. Target is L4 24 GB.", gpu_mem_gb
        )

    loaded: dict[str, tuple[Any, Any]] = {}
    for model_cfg in OSM_MODELS:
        try:
            model, tokenizer = load_model(model_cfg)
            loaded[model_cfg["name"]] = (model, tokenizer)
        except Exception as exc:
            logger.error("FAILED to load model %s: %s", model_cfg["name"], exc)
            raise

    logger.info("All %d OSM models loaded.", len(loaded))
    return loaded


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = load_all_osm_models()
    logger.info("Loaded models: %s", list(models.keys()))
