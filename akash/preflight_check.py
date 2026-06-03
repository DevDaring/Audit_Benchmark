"""Pre-flight check: verifies all fixes and readiness before dry run."""
import sys
import os
import subprocess

sys.path.insert(0, "/home/koushikdeb2009/Audit_Benchmark/Code/mirage")

PASS = []
FAIL = []


def chk(name, ok, detail=""):
    sym = "PASS" if ok else "FAIL"
    msg = "  [{}] {}".format(sym, name)
    if detail:
        msg += " -- " + detail
    print(msg)
    (PASS if ok else FAIL).append(name)


print("=" * 60)
print("MIRAGE PRE-FLIGHT CHECKS")
print("=" * 60)

# Git commit hash
r = subprocess.run(
    ["git", "-C", "/home/koushikdeb2009/Audit_Benchmark", "log", "--oneline", "-3"],
    capture_output=True,
    text=True,
)
print("Recent commits:")
for line in r.stdout.strip().splitlines():
    print("  " + line)

# 1. Phi-4-mini rope_scaling patch
try:
    from GPU_CPU.load_osm import _patch_transformers_compat  # noqa: applies patch
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        cache_dir="/home/koushikdeb2009/hf_cache",
    )
    chk("Phi4Mini_rope_patch", True, str(cfg.num_hidden_layers) + " layers")
except Exception as e:
    chk("Phi4Mini_rope_patch", False, str(e)[:120])

# 2. nnsight _nnsight_layer_proxies helper
try:
    from GPU_CPU.utils_attention import _nnsight_layer_proxies
    chk("nnsight_layer_proxies", True, "function present")
except Exception as e:
    chk("nnsight_layer_proxies", False, str(e)[:120])

# 3. All env keys present
from config import validate_all_keys
missing = validate_all_keys()
chk("ENV_KEYS", len(missing) == 0,
    "missing=" + str(missing) if missing else "all present")

# 4. Pentad dataset complete
import pandas as pd
from config import SEEDS_DIR, RESULTS_DIR
try:
    df = pd.read_parquet(SEEDS_DIR / "pentad_dataset.parquet")
    n_total = len(df)
    n_d = int((df["slot"] == "d").sum())
    n_e = int((df["slot"] == "e").sum())
    ok = n_total == 8016 and n_d == 1336 and n_e == 2004
    chk("Pentad_8016rows", ok,
        str(n_total) + " rows  d=" + str(n_d) + " e=" + str(n_e))
except Exception as e:
    chk("Pentad_8016rows", False, str(e)[:80])

# 5. Results dir clean (no prior production results)
beh = RESULTS_DIR / "behavioral_results.parquet"
cdva = RESULTS_DIR / "cdva_results.parquet"
both_clean = not beh.exists() and not cdva.exists()
chk("ResultsDir_CLEAN", both_clean,
    "clean - no prior results" if both_clean else
    "WARNING: existing results found - main run will resume from checkpoint")

# 6. CUDA + VRAM
import torch
if torch.cuda.is_available():
    dev = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    chk("CUDA", True, dev)
    chk("VRAM_40GB", gb >= 38, "{:.1f} GB".format(gb))
else:
    chk("CUDA", False, "no GPU detected")
    chk("VRAM_40GB", False, "no GPU")

# 7. Flash attention
try:
    import flash_attn
    chk("flash_attn", True, "v" + flash_attn.__version__)
except Exception as e:
    chk("flash_attn", False, str(e)[:80])

# 8. TransformerLens
try:
    import transformer_lens
    chk("transformer_lens", True, "installed")
except Exception as e:
    chk("transformer_lens", False, str(e)[:80])

# 9. nnsight
try:
    import nnsight
    chk("nnsight", True, "v" + str(nnsight.__version__))
except Exception as e:
    chk("nnsight", False, str(e)[:80])

# 10. Fallback providers in generators
try:
    from Dataset.context_shift_drafter import _GEMINI_BASE_URL, _MISTRAL_BASE_URL
    from Dataset.cot_attack_generator import _GEMINI_BASE_URL as g2
    ok = "google" in _GEMINI_BASE_URL and "mistral" in _MISTRAL_BASE_URL
    chk("API_fallback_cascade", ok,
        "DeepSeek->Gemini->Mistral configured")
except Exception as e:
    chk("API_fallback_cascade", False, str(e)[:80])

print()
print("RESULT: {} PASS  {} FAIL".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED CHECKS: " + str(FAIL))
    sys.exit(1)
else:
    print("ALL CHECKS PASSED - safe to start dry run")
