"""
File: TIST/patch_core.py
Purpose: A patching core general enough for the whole E1 battery.

GPU_CPU/utils_attention.patch_activation answers exactly one question: patch every layer
at the protected position and return the change in one option's logit. The E1 battery
needs to vary three things that function fixes:

  * WHICH LAYERS are patched            (E1.2 layer sweep)
  * WHICH POSITION supplies and receives the activation  (E1.1 placebo controls)
  * WHICH METRIC reads the result       (E1.6 metric robustness)

So this module exposes one primitive, `patched_logits`, that patches a chosen set of
layers from a chosen source position into a chosen target position and returns the full
final-position logit vector. Every metric in the battery is then a pure function of two
logit vectors, which means the three metrics of E1.6 cost nothing extra: they are read
from the same forward pass.

Tokenisation and position finding are imported from GPU_CPU.utils_attention rather than
reimplemented, so the pairs this module scores are the same pairs the production audit
scored.

Implements / builds on / cites:
  - Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022.
  - Zhang & Nanda (2024). "Towards Best Practices of Activation Patching in Language
    Models." ICLR 2024 -- noising and denoising directions, metric choice.
  - Vig et al. (2020). "Investigating Gender Bias in Language Models Using Causal
    Mediation Analysis." NeurIPS 2020 -- the indirect-effect decomposition of E1.5.

Part of the audit codebase (MIRAGE, TIST resubmission).
"""

import logging
from typing import Any

import numpy as np

from GPU_CPU.utils_attention import (
    _ensure_hooked_transformer,
    _ensure_nnsight_model,
    _get_token_position,
    _nnsight_layer_proxies,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive: run `prompt` with `layers` patched at `tgt_pos` from `src_acts`
# ---------------------------------------------------------------------------
def _resid_cache_tl(tl_model: Any, prompt: str) -> dict[int, Any]:
    import torch

    with torch.no_grad():
        _, cache = tl_model.run_with_cache(prompt, return_type=None)
    out = {}
    for layer in range(tl_model.cfg.n_layers):
        key = f"blocks.{layer}.hook_resid_post"
        if key in cache:
            out[layer] = cache[key][0].clone()      # [pos, d_model]
    return out


def _final_logits_tl(tl_model: Any, prompt: str, patch: dict | None) -> "np.ndarray":
    """
    Final-position logits for `prompt`. When `patch` is given it must carry
    {"acts": {layer: vector}, "tgt_pos": int}; those layers are overwritten at that
    position during the forward pass.
    """
    import torch

    if patch is None:
        with torch.no_grad():
            logits = tl_model(prompt, return_type="logits")
        return logits[0, -1, :].float().cpu().numpy()

    hooks = []
    tgt = patch["tgt_pos"]
    for layer, vec in patch["acts"].items():
        key = f"blocks.{layer}.hook_resid_post"

        def make(v: Any):
            def fn(value: Any, hook: Any) -> Any:
                if tgt < value.shape[1]:
                    value[0, tgt, :] = v.to(value.device, value.dtype)
                return value

            return fn

        hooks.append((key, make(vec)))

    with torch.no_grad():
        logits = tl_model.run_with_hooks(prompt, return_type="logits", fwd_hooks=hooks)
    return logits[0, -1, :].float().cpu().numpy()


def _resid_cache_nnsight(nn_model: Any, hf_model: Any, prompt: str) -> dict[int, Any]:
    layers_proxy, _ = _nnsight_layer_proxies(nn_model, hf_model)
    saved = {}
    with nn_model.trace(prompt):
        for i in range(len(layers_proxy)):
            out = layers_proxy[i].output
            saved[i] = (out[0] if isinstance(out, tuple) else out).save()
    return {i: v.value[0].clone() for i, v in saved.items()}


def _final_logits_nnsight(nn_model: Any, hf_model: Any, prompt: str, patch: dict | None) -> "np.ndarray":
    layers_proxy, lm_head = _nnsight_layer_proxies(nn_model, hf_model)
    with nn_model.trace(prompt):
        if patch is not None:
            tgt = patch["tgt_pos"]
            for layer, vec in patch["acts"].items():
                out = layers_proxy[layer].output
                hidden = out[0] if isinstance(out, tuple) else out
                hidden[0, tgt, :] = vec
        logits = lm_head.output.save()
    arr = logits.value
    return arr[0, -1, :].float().cpu().numpy()


class Patcher:
    """Uniform patching interface over TransformerLens and NNsight."""

    def __init__(self, model: Any, tokenizer: Any, patching_lib: str):
        self.tokenizer = tokenizer
        self.lib = "nnsight" if "nnsight" in str(patching_lib).lower() else "transformer_lens"
        if self.lib == "nnsight":
            self.hf_model = model
            self.model = _ensure_nnsight_model(model, tokenizer)
            layers_proxy, _ = _nnsight_layer_proxies(self.model, model)
            self.n_layers = len(layers_proxy)
        else:
            self.model = _ensure_hooked_transformer(model, tokenizer)
            self.n_layers = self.model.cfg.n_layers

    # -- caching ----------------------------------------------------------
    def cache(self, prompt: str) -> dict[int, Any]:
        if self.lib == "nnsight":
            return _resid_cache_nnsight(self.model, self.hf_model, prompt)
        return _resid_cache_tl(self.model, prompt)

    # -- forward ----------------------------------------------------------
    def logits(self, prompt: str, patch: dict | None = None) -> "np.ndarray":
        if self.lib == "nnsight":
            return _final_logits_nnsight(self.model, self.hf_model, prompt, patch)
        return _final_logits_tl(self.model, prompt, patch)

    def patched_logits(
        self,
        src_cache: dict[int, Any],
        src_pos: int,
        tgt_prompt: str,
        tgt_pos: int,
        layers: list[int] | None = None,
    ) -> "np.ndarray":
        """Patch `layers` (default: all) at `tgt_pos` from `src_cache[layer][src_pos]`."""
        chosen = range(self.n_layers) if layers is None else layers
        acts = {}
        for layer in chosen:
            if layer in src_cache and src_pos < src_cache[layer].shape[0]:
                acts[layer] = src_cache[layer][src_pos].clone()
        if not acts:
            return self.logits(tgt_prompt)
        return self.logits(tgt_prompt, {"acts": acts, "tgt_pos": tgt_pos})

    # -- token helpers ----------------------------------------------------
    def position_of(self, prompt: str, token: str) -> int | None:
        return _get_token_position(self.tokenizer, prompt, token)

    def token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def n_tokens(self, prompt: str) -> int:
        if self.lib == "nnsight":
            return len(self.tokenizer.encode(prompt))
        return int(self.model.to_tokens(prompt).shape[1])

    def decoded_tokens(self, prompt: str) -> list[str]:
        if self.lib == "nnsight":
            ids = self.tokenizer.encode(prompt)
        else:
            ids = self.model.to_tokens(prompt)[0].tolist()
        return [self.tokenizer.decode([i]) for i in ids]


# ---------------------------------------------------------------------------
# Metrics (E1.6). Each takes the unpatched and patched final-position logits.
# ---------------------------------------------------------------------------
def _first_id(patcher: Patcher, text: str) -> int | None:
    ids = patcher.token_ids(text)
    return ids[0] if ids else None


def metric_single_logit(base: "np.ndarray", patched: "np.ndarray", tok_id: int) -> float:
    """The production metric: change in one option's logit."""
    return float(patched[tok_id] - base[tok_id])


def metric_logit_diff(
    base: "np.ndarray", patched: "np.ndarray", id_pro: int, id_anti: int
) -> float:
    """Change in the stereotype-minus-antistereotype logit difference."""
    return float((patched[id_pro] - patched[id_anti]) - (base[id_pro] - base[id_anti]))


def metric_kl(base: "np.ndarray", patched: "np.ndarray") -> float:
    """KL(patched || base) over the full next-token distribution, in nats."""
    def _logsoftmax(x: "np.ndarray") -> "np.ndarray":
        m = x.max()
        return x - m - np.log(np.exp(x - m).sum())

    lp, lb = _logsoftmax(patched.astype(np.float64)), _logsoftmax(base.astype(np.float64))
    p = np.exp(lp)
    return float((p * (lp - lb)).sum())


# ---------------------------------------------------------------------------
# Placebo position choosers (E1.1)
# ---------------------------------------------------------------------------
_FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for",
    "with", "is", "was", "were", "are", "be", "been", "that", "this", "it", "as",
}


def choose_placebo_positions(
    patcher: Patcher, prompt: str, protected_pos: int, rng: "np.random.Generator"
) -> dict[str, int | None]:
    """
    Two within-prompt placebo positions:

      content_pos  -- a random alphabetic token that is not the protected token and not
                      a function word. Tests whether any content position moves the logit.
      function_pos -- a random function word. Tests whether the effect is carried by
                      position occupancy rather than by content.
    """
    toks = patcher.decoded_tokens(prompt)
    content, function = [], []
    for i, t in enumerate(toks):
        s = t.strip().lower()
        if not s or not s.isalpha() or i == protected_pos:
            continue
        (function if s in _FUNCTION_WORDS else content).append(i)

    return {
        "content_pos": int(rng.choice(content)) if content else None,
        "function_pos": int(rng.choice(function)) if function else None,
    }
