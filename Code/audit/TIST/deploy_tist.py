"""
File: TIST/deploy_tist.py
Purpose: Provision the Akash GPU lease for the TIST resubmission battery.

Thin wrapper over GPU_Remaining/deploy_akash.py. That module already implements the
Console Managed-Wallet flow correctly (create deployment, wait for bids, take the
cheapest, sign the lease, save state, close). Rather than fork 160 lines, this file
reuses its functions and overrides four module globals:

  ENV       -> Code/audit/.env, which is where Bharat_AKASH_Key lives
  TEMPLATE  -> TIST/sdl_tist.yaml
  STATE     -> TIST/.deploy_state.json
  key name  -> Bharat_AKASH_Key, mapped onto the AKASH_API_KEY name the wrapped
               module expects. The two keys are different accounts; the TIST run uses
               the Bharat one on Koushik's instruction.

GPU preference is restricted to cards with 40 GB or more, ordered cheapest-first.

24 GB cards are excluded deliberately. The models are 8B or smaller in bfloat16, so the
weights fit, but TransformerLens holds a converted HookedTransformer alongside the loaded
HF model and caches the residual stream at every layer during patching. Llama-3.1-8B on a
24 GB card is close enough to the limit that an out-of-memory failure part-way through a
paid run is a real possibility, and the original CDVA results were produced on an A100
40 GB. Paying more per hour is cheaper than losing a model's results to an OOM.

Usage:
  python TIST/deploy_tist.py --deposit 20 --wait 180
  python TIST/deploy_tist.py --status
  python TIST/deploy_tist.py --close
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
sys.path.insert(0, str(AUDIT))
sys.path.insert(0, str(AUDIT / "GPU_Remaining"))

import deploy_akash as da  # noqa: E402

# Point the wrapped module at this run's files.
da.ENV = AUDIT / ".env"
da.TEMPLATE = HERE / "sdl_tist.yaml"
da.STATE = HERE / ".deploy_state.json"

# 40 GB and above only, cheapest tier first. No 24 GB card appears here on purpose.
DEFAULT_GPUS = "a6000,a40,l40s,a100,h100,h200"

# Guard against a 24 GB card slipping back in through --gpus.
_MIN_VRAM_GB = 40
_VRAM_GB = {
    "rtx4090": 24, "rtx4080": 16, "rtx3090": 24, "l4": 24, "a10": 24, "a10g": 24,
    "l40": 48, "l40s": 48, "a6000": 48, "a40": 48, "a100": 40, "h100": 80, "h200": 141,
}


def load_env() -> dict:
    env = da.read_env()
    key = env.get("Bharat_AKASH_Key", "").strip()
    if not key:
        print("Bharat_AKASH_Key missing from Code/audit/.env")
        sys.exit(1)
    env["AKASH_API_KEY"] = key
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", type=float, default=20.0)
    ap.add_argument("--max-price", type=int, default=100000)
    ap.add_argument("--wait", type=int, default=150)
    ap.add_argument("--gpus", default=DEFAULT_GPUS)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--close", action="store_true")
    args = ap.parse_args()

    env = load_env()

    if args.status:
        da.show_status(env)
        return
    if args.close:
        da.close(env)
        return

    requested = [g.strip() for g in args.gpus.split(",") if g.strip()]
    too_small = [g for g in requested if _VRAM_GB.get(g, _MIN_VRAM_GB) < _MIN_VRAM_GB]
    if too_small:
        print(
            f"refusing {', '.join(too_small)}: under the {_MIN_VRAM_GB} GB floor. "
            "TransformerLens holds a converted model alongside the HF weights and caches "
            "the residual stream at every layer, so Llama-3.1-8B risks an out-of-memory "
            "failure part-way through a paid run."
        )
        sys.exit(2)

    for gpu in requested:
        state = da.provision(env, gpu, args.deposit, args.max_price, args.wait)
        if state:
            print(json.dumps(state, indent=2))
            print("\nlease is up. bootstrap_tist.sh runs automatically inside the container.")
            print("watch progress with: python TIST/deploy_tist.py --status")
            print("results land in Code/audit/results/tist/ and are pushed every 15 minutes.")
            return
    print("no provider accepted any of the requested GPU models")
    sys.exit(2)


if __name__ == "__main__":
    main()
