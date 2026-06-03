"""
File: GPU_CPU/utils_attention.py
Purpose: Uniform interface for causal activation patching across
         TransformerLens (Llama, Gemma) and nnsight (Qwen, Phi-4).

TransformerLens note:
  load_osm.py loads plain HF AutoModelForCausalLM objects.  The patching
  functions here convert them to the appropriate patching-library wrapper
  on-demand and cache the result so conversion happens once per model.

nnsight note (v0.6+):
  After the first trace exits, saved proxies must be accessed via .value
  before being used inside a second trace context.  Failing to do so
  raises a RuntimeError in nnsight 0.6+.

Implements / builds on / cites:
  - Meng et al. (2022). "Locating and Editing Factual Associations in GPT."
    NeurIPS 2022. https://arxiv.org/abs/2202.05262 -- activation patching.
  - Pearl (2009). Causality. Cambridge University Press.
    -- do-calculus / interventional framing.
  - TransformerLens: Nanda & Bloom (2022). https://github.com/neelnanda-io/TransformerLens
  - nnsight: Fiotto-Kaufman et al. (2023). https://github.com/ndif-team/nnsight

Part of the MIRAGE codebase. See README.md for full project context.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache: HF model id -> HookedTransformer
# Populated lazily so we only convert once per model per process.
# ---------------------------------------------------------------------------
_TL_MODEL_CACHE: dict[str, Any] = {}


def _get_token_position(tokenizer: Any, prompt: str, target_token: str) -> int | None:
    """
    Find the position of the first occurrence of target_token in the
    tokenised prompt using a deterministic, tokenizer-aware scan.

    Returns the token index, or None if not found.
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    target_lower = target_token.lower()
    for i, tok_str in enumerate(token_strs):
        if target_lower in tok_str.lower():
            return i
    return None


# ---------------------------------------------------------------------------
# TransformerLens helpers
# ---------------------------------------------------------------------------

def _ensure_hooked_transformer(model: Any, tokenizer: Any) -> Any:
    """
    Return a HookedTransformer wrapping the given model.

    If `model` is already a HookedTransformer, return it unchanged.
    Otherwise create one from the HF model in-place (weights are shared
    via `hf_model=` parameter -- no extra VRAM copy).

    The result is cached by model HF ID so conversion happens once per
    process.  fold_ln / center_writing_weights are disabled so logits
    match the HF model exactly (required for valid delta_logit comparisons).
    """
    try:
        import transformer_lens  # type: ignore
        if isinstance(model, transformer_lens.HookedTransformer):
            return model
    except ImportError as exc:
        raise ImportError(
            "transformer_lens not installed. "
            "Install with: pip install transformer_lens==2.18.0"
        ) from exc

    import torch

    hf_id = getattr(model.config, "_name_or_path", "") or "unknown_model"

    if hf_id in _TL_MODEL_CACHE:
        return _TL_MODEL_CACHE[hf_id]

    logger.info(
        "Converting HF model '%s' to HookedTransformer for CDVA patching ...", hf_id
    )

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    tl_model = transformer_lens.HookedTransformer.from_pretrained(
        hf_id,
        hf_model=model,
        dtype=dtype,
        # Preserve raw HF weights -- no folding or centering.
        # Folding LayerNorm into preceding weights changes the numeric scale,
        # making delta_logit values incomparable across patched/unpatched runs.
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        # Respect the HUGGINGFACE_TOKEN already used when loading the HF model
        move_to_device=True,
        device=str(device),
    )
    tl_model.eval()

    _TL_MODEL_CACHE[hf_id] = tl_model
    logger.info("HookedTransformer for '%s' cached (device=%s).", hf_id, device)
    return tl_model


# ---------------------------------------------------------------------------
# TransformerLens patching (Llama, Gemma)
# ---------------------------------------------------------------------------

def patch_activation_transformer_lens(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
) -> float:
    """
    Causal activation patch using TransformerLens.

    Accepts either a HookedTransformer or a plain HF AutoModelForCausalLM.
    Plain HF models are converted to HookedTransformer on first call and
    cached (see _ensure_hooked_transformer).

    For each layer, replaces the residual-stream activation at position_b
    in prompt_B with the cached activation at position_a from prompt_A.

    Returns
    -------
    float
        delta_logit = logit_patched(bias_answer) - logit_original(bias_answer)

    Implements:
        Pearl (2009) do-calculus intervention:
            do(activation_{L,b} := activation_{L,a})
    """
    import torch

    tl_model = _ensure_hooked_transformer(model, tokenizer)

    # Tokenise
    tokens_b = tl_model.to_tokens(prompt_b)

    # Forward pass on prompt_A to cache residual activations
    with torch.no_grad():
        _, cache_a = tl_model.run_with_cache(prompt_a, return_type=None)

    n_layers = tl_model.cfg.n_layers

    # Build patching hooks: replace resid_post at position_b with cache_a at position_a
    hooks = []
    for layer in range(n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key not in cache_a:
            continue
        cached_act = cache_a[key][0, position_a, :].clone()

        def make_hook(act: "torch.Tensor") -> Any:
            def hook_fn(value: "torch.Tensor", hook: Any) -> "torch.Tensor":
                value[0, position_b, :] = act
                return value
            return hook_fn

        hooks.append((key, make_hook(cached_act)))

    # Forward pass on prompt_B with hooks (patched)
    with torch.no_grad():
        logits_patched = tl_model.run_with_hooks(prompt_b, fwd_hooks=hooks)

    # Forward pass on prompt_B without hooks (original)
    with torch.no_grad():
        logits_original = tl_model(prompt_b)

    # Find logit for bias_answer token
    bias_token_ids = tl_model.to_tokens(bias_answer, prepend_bos=False)[0]
    if len(bias_token_ids) == 0:
        logger.warning("Could not tokenise bias_answer '%s'.", bias_answer)
        return 0.0

    bias_tok = bias_token_ids[0].item()
    last_pos = -1
    logit_patched = logits_patched[0, last_pos, bias_tok].item()
    logit_original = logits_original[0, last_pos, bias_tok].item()
    return float(logit_patched - logit_original)


# ---------------------------------------------------------------------------
# nnsight patching (Qwen, Phi-4)
# ---------------------------------------------------------------------------

# Module-level LanguageModel cache for nnsight — prevents re-wrapping the same
# HF model on every seed call (which leaks GPU memory and slows CDVA by ~2×).
_NNSIGHT_MODEL_CACHE: dict[str, Any] = {}


def _ensure_nnsight_model(model: Any, tokenizer: Any) -> Any:
    """Return a cached nnsight LanguageModel wrapper for the given HF model."""
    try:
        from nnsight import LanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError("nnsight not installed.") from exc

    hf_id = getattr(model.config, "_name_or_path", "") or id(model)
    key = str(hf_id)
    if key not in _NNSIGHT_MODEL_CACHE:
        logger.info("Creating nnsight LanguageModel wrapper for '%s' (cached).", hf_id)
        _NNSIGHT_MODEL_CACHE[key] = LanguageModel(model, tokenizer=tokenizer)
    return _NNSIGHT_MODEL_CACHE[key]


def _nnsight_layer_proxies(nn_model: Any, hf_model: Any) -> tuple[Any, Any]:
    """
    Return (layers_proxy, lm_head_proxy) for the given nnsight LanguageModel.

    HuggingFace CausalLM classes come in two shapes:

      Shape A — layers directly on top-level class (rare):
          hf_model.layers        → decoder stack
          hf_model.lm_head       → vocab projection

      Shape B — layers nested inside an inner .model attribute (Qwen2, Phi3/4,
                LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM, …):
          hf_model.model.layers  → decoder stack
          hf_model.lm_head       → vocab projection (always at top level)

    We detect the shape from the *real* (non-proxy) HF model and return the
    correct nnsight proxy paths so the trace assignment works for all models.
    """
    inner = getattr(hf_model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        # Shape B: Qwen2ForCausalLM, Phi3ForCausalLM, LlamaForCausalLM, …
        layers_proxy = nn_model.model.model.layers
        lm_head_proxy = nn_model.model.lm_head
        logger.debug(
            "nnsight layer path: .model.model.layers (inner model) "
            "for %s", type(hf_model).__name__,
        )
    elif hasattr(hf_model, "layers"):
        # Shape A: direct .layers (some GPT-NeoX style models)
        layers_proxy = nn_model.model.layers
        lm_head_proxy = nn_model.model.lm_head
        logger.debug(
            "nnsight layer path: .model.layers (top-level) "
            "for %s", type(hf_model).__name__,
        )
    else:
        raise AttributeError(
            f"Cannot locate decoder layers in {type(hf_model).__name__}. "
            "Expected .model.layers or .layers on the CausalLM object."
        )
    return layers_proxy, lm_head_proxy


def patch_activation_nnsight(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
) -> float:
    """
    Causal activation patch using nnsight (v0.3.7+).

    Replaces residual stream at (layer, position_b) in prompt_B with the
    cached residual at (layer, position_a) from prompt_A, for every layer.

    Architecture note:
      HF CausalLM classes (Qwen2, Phi3/4, Llama) nest their decoder layers
      at .model.layers inside the outer CausalLM wrapper; the lm_head lives
      at the outer level.  We detect this dynamically via _nnsight_layer_proxies
      to avoid hard-coding per-model paths.

    nnsight note:
      After a trace context exits, saved proxies are resolved — their tensor
      value is accessible via `.value`.  When assigning a saved activation
      inside a *second* trace context, we must pass `.value` explicitly;
      passing the proxy object itself raises RuntimeError.

    Returns
    -------
    float
        delta_logit
    """
    import torch

    nn_model = _ensure_nnsight_model(model, tokenizer)
    layers_proxy, lm_head_proxy = _nnsight_layer_proxies(nn_model, model)

    # ------------------------------------------------------------------
    # Pass 1: collect residual activations from prompt_A
    # ------------------------------------------------------------------
    cache_a_proxies: dict[int, Any] = {}
    with nn_model.trace(prompt_a):
        for layer_idx, layer in enumerate(layers_proxy):
            cache_a_proxies[layer_idx] = layer.output[0][:, position_a, :].save()

    # Resolve .value OUTSIDE the trace so we have concrete tensors
    cache_a_vals: dict[int, "torch.Tensor"] = {
        idx: proxy.value.clone() for idx, proxy in cache_a_proxies.items()
    }

    # ------------------------------------------------------------------
    # Pass 2: patched forward on prompt_B (inject cache_a_vals)
    # ------------------------------------------------------------------
    with nn_model.trace(prompt_b):
        for layer_idx, layer in enumerate(layers_proxy):
            if layer_idx in cache_a_vals:
                # Use the concrete tensor (.value already resolved) --
                # NOT the proxy object, which is invalid in a new trace.
                layer.output[0][:, position_b, :] = cache_a_vals[layer_idx]
        patched_logits = lm_head_proxy.output.save()

    # ------------------------------------------------------------------
    # Pass 3: unpatched forward on prompt_B (baseline)
    # ------------------------------------------------------------------
    with nn_model.trace(prompt_b):
        original_logits = lm_head_proxy.output.save()

    bias_token_ids = tokenizer.encode(bias_answer, add_special_tokens=False)
    if not bias_token_ids:
        logger.warning("Could not tokenise bias_answer '%s'.", bias_answer)
        return 0.0

    bias_tok = bias_token_ids[0]
    logit_patched = patched_logits.value[0, -1, bias_tok].item()
    logit_original = original_logits.value[0, -1, bias_tok].item()
    return float(logit_patched - logit_original)


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def patch_activation(
    model: Any,
    tokenizer: Any,
    prompt_a: str,
    prompt_b: str,
    position_a: int,
    position_b: int,
    bias_answer: str,
    patching_lib: str,
) -> float:
    """
    Dispatch to the appropriate patching library.

    Parameters
    ----------
    model : Any
        Loaded model object (plain HF AutoModelForCausalLM or HookedTransformer).
        For transformer_lens path, the model is auto-converted if needed.
    tokenizer : Any
        Corresponding tokenizer.
    prompt_a : str
        Source prompt (activation source).
    prompt_b : str
        Target prompt (to be patched).
    position_a : int
        Demographic-token position in prompt_A.
    position_b : int
        Demographic-token position in prompt_B.
    bias_answer : str
        The answer token whose logit shift is measured.
    patching_lib : str
        'transformer_lens' or 'nnsight'.

    Returns
    -------
    float
        delta_logit (patched - original).
    """
    if patching_lib == "transformer_lens":
        return patch_activation_transformer_lens(
            model, tokenizer, prompt_a, prompt_b, position_a, position_b, bias_answer
        )
    elif patching_lib == "nnsight":
        return patch_activation_nnsight(
            model, tokenizer, prompt_a, prompt_b, position_a, position_b, bias_answer
        )
    else:
        raise ValueError(
            f"Unknown patching_lib: '{patching_lib}'. "
            "Use 'transformer_lens' or 'nnsight'."
        )
