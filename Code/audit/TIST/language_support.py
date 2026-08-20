"""
File: TIST/language_support.py
Purpose: Decide which (model, language) pairs may be audited at all.

Why this exists. A bias audit in language L asks whether the model's answer moves with a
protected attribute. That question is only meaningful if the model processes L in the
first place. Run it on a model that was never built for L and a low score conflates two
different things: "this model is biased" and "this model cannot read the prompt". The
resulting number is not a fairness measurement, it is a language-competence measurement
wearing a fairness label.

That is the same construct-validity failure this paper documents in behavioural bias
benchmarks, so committing it in the paper's own multilingual extension would be
indefensible. Coverage is therefore gated on the vendor's declared support, and an
ungated pair is reported as "not applicable", never as a failure or a zero.

The trigger was concrete. Gemma-2-2B is documented by Google as English-only, yet the
first run located 3,433 of 3,433 Bengali counterfactual positions. Position location only
means the tokeniser found the swapped token; it says nothing about comprehension. Without
this gate that 100% would have been read as Bengali coverage.

Declared support, from the vendors' own model cards (accessed 14 August 2026):

  Llama-3.1-8B-Instruct  Meta lists exactly eight supported languages: English, German,
                         French, Italian, Portuguese, Hindi, Spanish, Thai. Hindi yes,
                         Bengali no.
                         https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md
  Gemma-2-2B-it          Google's Gemma 2 model card describes the models as English.
                         Neither Hindi nor Bengali.
                         https://ai.google.dev/gemma/docs/core/model_card_2
  Phi-4-mini-instruct    Microsoft lists 23 languages; the list contains neither Hindi
                         nor Bengali.
                         https://huggingface.co/microsoft/Phi-4-mini-instruct
  Qwen2.5-7B-Instruct    Alibaba claims support for "over 29 languages" and names a
                         subset that includes neither Hindi nor Bengali. Absent an
                         explicit claim, this module treats them as unsupported, because
                         the burden is on the declared claim rather than on inference.
                         https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Consequence for the study as currently configured: Hindi is auditable on Llama-3.1-8B
only, and Bengali on no open model in the set. Extending the multilingual claim needs a
model that declares the language, for example Gemma 3, which Google documents for over
140 languages.

Part of the MIRAGE audit codebase.
"""

# language -> models whose vendor documentation declares support
DECLARED_SUPPORT: dict[str, set[str]] = {
    "en": {
        "llama-3.1-8b-instruct",
        "qwen2.5-7b-instruct",
        "gemma-2-2b-it",
        "phi-4-mini-instruct",
    },
    "hi": {
        "llama-3.1-8b-instruct",
    },
    "bn": set(),
}

# Shown in the paper and in the logs, so a skip is never mistaken for a failure.
REASON: dict[tuple[str, str], str] = {
    ("qwen2.5-7b-instruct", "hi"): "Qwen2.5 names 29 languages; Hindi is not among them",
    ("qwen2.5-7b-instruct", "bn"): "Qwen2.5 names 29 languages; Bengali is not among them",
    ("gemma-2-2b-it", "hi"): "Gemma 2 is documented as English",
    ("gemma-2-2b-it", "bn"): "Gemma 2 is documented as English",
    ("phi-4-mini-instruct", "hi"): "Phi-4-mini lists 23 languages; Hindi is not among them",
    ("phi-4-mini-instruct", "bn"): "Phi-4-mini lists 23 languages; Bengali is not among them",
    ("llama-3.1-8b-instruct", "bn"): "Llama 3.1 lists 8 languages; Bengali is not among them",
}


def is_supported(model_name: str, lang: str, results_dir=None) -> bool:
    """
    May this (model, language) pair be audited?

    Measurement wins over the label. A vendor list is a conservative legal document, not a
    capability measurement: Qwen2.5 names 29 languages without listing Hindi yet may well
    process it, and excluding on the label alone throws away real coverage.

    So a measured verdict from TIST/competence_probe decides whenever one exists, in either
    direction. It can admit a model the vendor never listed, and it can exclude one the
    vendor did list but which fails the probe. The declared list is only the fallback for
    pairs that have not been probed yet.
    """
    if results_dir is not None:
        from TIST.competence_probe import load as _load_competence

        verdict = _load_competence(results_dir).get(f"{model_name}|{lang}")
        if verdict is not None:
            return bool(verdict.get("competent"))

    return model_name in DECLARED_SUPPORT.get(lang, set())


def skip_reason(model_name: str, lang: str, results_dir=None) -> str:
    """Why a pair is out of scope. Used in logs and in the not-applicable cells."""
    if results_dir is not None:
        from TIST.competence_probe import load as _load_competence

        verdict = _load_competence(results_dir).get(f"{model_name}|{lang}")
        if verdict is not None and not verdict.get("competent"):
            return f"failed the language-competence probe: {verdict.get('reason')}"

    return REASON.get(
        (model_name, lang),
        f"{model_name} does not declare support for '{lang}'",
    )


def auditable_pairs(model_names, langs) -> list[tuple[str, str]]:
    return [(m, l) for l in langs for m in model_names if is_supported(m, l)]
