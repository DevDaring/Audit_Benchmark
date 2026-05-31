"""
akash/deploy.py — Deploy a MIRAGE GPU VM on Akash Network.

Usage (one-time manual steps described below, then run this script):

  1. Install the Akash CLI:  https://akash.network/docs/deployments/akash-cli/overview/
     OR use the Akash Console web UI:  https://console.akash.network/

  2. Fund your wallet with AKT (or use the Akash Console funded account).

  3. Set env vars (or add to .env):
       AKASH_KEY_NAME          <your wallet key name, e.g. "mirage_key">
       AKASH_KEYRING_BACKEND   os   (or "test" for quick experiments)
       AKASH_NODE              https://rpc.akash.network:443
       AKASH_CHAIN_ID          akashnet-2
       AKASH_FROM              <your wallet address>
       GITHUB_TOKEN            <your GitHub token (optional, for private repos)>

  4. Run:  python akash/deploy.py

This script:
  - Generates the SDL (Stack Definition Language) YAML for an A100 40 GB deployment
  - Optionally submits it via `akash tx deployment create` (CLI path)
  - Prints the manual Console-UI instructions if CLI is not on PATH

SDL approach used here (Console-compatible):
  https://akash.network/docs/deployments/akash-cli/deployment-with-sdl/

Note on provider selection:
  A100 40 GB providers on Akash (as of May 2026):
    - provider4.us-east.akash.pub          (GPUHive, TX)
    - provider2.eu-west.akashprovid.com    (Zondax, EU)
    - gpu.akash.pub                        (general pool)
  This script bids the minimum and lets Akash find the cheapest provider.
"""

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# ------------------------------------------------------------------ config
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / "Code" / "mirage" / ".env"
SDL_OUT = Path(__file__).parent / "deployment.yaml"

# A100 40 GB SDL template.  Adjust GITHUB_REPO and ROOT_PASSWORD as needed.
SDL_TEMPLATE = dedent("""\
---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
      - MIRAGE_ENV_B64={env_b64}
    command:
      - bash
      - -c
      - |
        apt-get update -qq && apt-get install -y git curl tmux openssh-server > /dev/null
        mkdir -p /workspace
        cd /workspace
        git clone $GITHUB_REPO Audit_Benchmark
        bash Audit_Benchmark/akash/startup.sh
    expose:
      - port: 22
        as: 22
        to:
          - global: true

profiles:
  compute:
    mirage:
      resources:
        cpu:
          units: 16
        memory:
          size: 80Gi
        storage:
          - size: 200Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
                - model: a100
                  ram: 80Gi
  placement:
    akash:
      pricing:
        mirage:
          denom: uakt
          amount: 1000000

deployment:
  mirage:
    akash:
      profile: mirage
      count: 1
""")


def _load_env_as_b64() -> str:
    """Base64-encode the .env file so it can be injected via SDL env var."""
    if not ENV_FILE.exists():
        print(f"[deploy] WARNING: {ENV_FILE} not found — .env will be missing on VM.")
        return ""
    raw = ENV_FILE.read_bytes()
    return base64.b64encode(raw).decode()


def _write_sdl(env_b64: str) -> Path:
    sdl = SDL_TEMPLATE.format(env_b64=env_b64)
    SDL_OUT.write_text(sdl, encoding="utf-8")
    print(f"[deploy] SDL written to: {SDL_OUT}")
    return SDL_OUT


def _try_cli_deploy(sdl_path: Path) -> bool:
    """Attempt deployment via akash CLI. Returns True on success."""
    if not shutil.which("akash"):
        return False

    key_name = os.environ.get("AKASH_KEY_NAME", "")
    if not key_name:
        print("[deploy] AKASH_KEY_NAME not set; skipping CLI deploy.")
        return False

    cmd = [
        "akash", "tx", "deployment", "create",
        str(sdl_path),
        "--from", key_name,
        "--node", os.environ.get("AKASH_NODE", "https://rpc.akash.network:443"),
        "--chain-id", os.environ.get("AKASH_CHAIN_ID", "akashnet-2"),
        "--gas", "auto",
        "--gas-adjustment", "1.4",
        "--keyring-backend", os.environ.get("AKASH_KEYRING_BACKEND", "os"),
        "-y",
    ]
    print("[deploy] Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def _print_console_instructions(sdl_path: Path) -> None:
    lines = [
        "",
        "=" * 68,
        "  AKASH CONSOLE DEPLOYMENT INSTRUCTIONS",
        "  (use these if you prefer the web UI over the CLI)",
        "=" * 68,
        "  1. Go to:  https://console.akash.network/",
        "  2. Connect your AKT wallet (Keplr or Leap)",
        '  3. Click "Deploy" -> "From SDL file"',
        f"  4. Upload:  {sdl_path}",
        "  5. Review resources: 1x A100 40 GB, 16 CPU, 80 GB RAM",
        "  6. Set deposit (2 AKT recommended) and click 'Deploy'",
        "  7. Wait for bids (30-60 s), select cheapest provider",
        "  8. Accept the lease -- container boots, SSH becomes available",
        "",
        "  SSH access:",
        "    ssh root@<PROVIDER_IP> -p <FORWARDED_PORT>",
        "    password: MirageVM2026!",
        "",
        "  After SSH:",
        "    tmux attach -t mirage",
        "    # or run: python akash/run_install.py --host <IP> --port <PORT>",
        "=" * 68,
        "",
    ]
    print("\n".join(lines))


def main() -> None:
    print("[deploy] Generating Akash SDL for MIRAGE A100 40 GB deployment ...")
    env_b64 = _load_env_as_b64()
    sdl_path = _write_sdl(env_b64)

    deployed = _try_cli_deploy(sdl_path)
    if not deployed:
        _print_console_instructions(sdl_path)

    print("[deploy] Done.")


if __name__ == "__main__":
    main()
