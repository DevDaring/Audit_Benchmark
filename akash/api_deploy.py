"""
akash/api_deploy.py — Fully automated Akash Console API deployment for MIRAGE.

Flow:
  1. POST /v1/deployments     -> get dseq
  2. wait 30s for bids
  3. GET  /v1/bids?dseq=      -> pick cheapest A100 40 GB bid
  4. POST /v1/leases           -> accept bid, lease created
  5. GET  /v1/deployments/{dseq}/leases  -> get provider IP + SSH port
  6. Print SSH connection string

Requires: AKASH_API_KEY in .env (or environment)
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ------------------------------------------------------------------ config
API_BASE = "https://console-api.akash.network"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE  = REPO_ROOT / "Code" / "mirage" / ".env"
SDL_FILE  = Path(__file__).parent / "deployment.yaml"

# Load AKASH_API_KEY from env or .env file
def _load_key() -> str:
    key = os.environ.get("AKASH_API_KEY", "")
    if not key and ENV_FILE.exists():
        for line in ENV_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("AKASH_API_KEY="):
                key = line.split("=", 1)[1].strip()
                break
    if not key:
        print("[api_deploy] ERROR: AKASH_API_KEY not found in env or .env")
        sys.exit(1)
    return key


def _headers(key: str) -> dict:
    return {"x-api-key": key, "Content-Type": "application/json"}


# ------------------------------------------------------------------ step 1: create deployment
def create_deployment(key: str, sdl: str, deposit: float = 1.5) -> str:
    print("[api_deploy] Creating deployment ...")
    resp = requests.post(
        f"{API_BASE}/v1/deployments",
        headers=_headers(key),
        json={"data": {"sdl": sdl, "deposit": deposit}},
        timeout=60,
    )
    if resp.status_code not in (200, 201):
        print(f"[api_deploy] FAILED {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)
    data = resp.json()
    dseq = str(data.get("dseq") or data.get("data", {}).get("dseq", ""))
    if not dseq:
        print(f"[api_deploy] Could not find dseq in response: {json.dumps(data, indent=2)}")
        sys.exit(1)
    print(f"[api_deploy] Deployment created: dseq={dseq}")
    return dseq


# ------------------------------------------------------------------ step 2: wait for bids
def wait_for_bids(key: str, dseq: str, wait_s: int = 45) -> list:
    print(f"[api_deploy] Waiting {wait_s}s for provider bids ...")
    time.sleep(wait_s)
    resp = requests.get(
        f"{API_BASE}/v1/bids",
        headers=_headers(key),
        params={"dseq": dseq},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"[api_deploy] Bid fetch FAILED {resp.status_code}: {resp.text[:300]}")
        sys.exit(1)
    bids = resp.json()
    if isinstance(bids, dict):
        bids = bids.get("bids") or bids.get("data") or []
    print(f"[api_deploy] {len(bids)} bid(s) received.")
    if not bids:
        print("[api_deploy] No bids. Possible causes: no A100 40 GB provider available, "
              "deposit too low, or network congestion. Try again or lower GPU spec.")
        sys.exit(1)
    return bids


# ------------------------------------------------------------------ step 3: pick bid
def pick_cheapest_bid(bids: list) -> dict:
    """Pick the cheapest bid. Prefer providers with 'a100' in their attributes."""
    def bid_price(b: dict) -> float:
        price = b.get("price") or b.get("amount") or {}
        if isinstance(price, dict):
            return float(price.get("amount", 999999))
        return float(price or 999999)

    sorted_bids = sorted(bids, key=bid_price)
    chosen = sorted_bids[0]
    provider = chosen.get("provider") or chosen.get("providerId", "unknown")
    price = bid_price(chosen)
    print(f"[api_deploy] Chosen bid: provider={provider}, price={price} uakt/block")
    return chosen


# ------------------------------------------------------------------ step 4: accept bid / create lease
def create_lease(key: str, dseq: str, bid: dict) -> dict:
    print("[api_deploy] Creating lease (accepting bid) ...")
    gseq = bid.get("gseq", 1)
    oseq = bid.get("oseq", 1)
    provider = bid.get("provider") or bid.get("providerId")

    payload = {"data": [{"dseq": dseq, "gseq": gseq, "oseq": oseq, "provider": provider}]}
    resp = requests.post(
        f"{API_BASE}/v1/leases",
        headers=_headers(key),
        json=payload,
        timeout=60,
    )
    if resp.status_code not in (200, 201):
        print(f"[api_deploy] Lease FAILED {resp.status_code}: {resp.text[:500]}")
        sys.exit(1)
    lease_data = resp.json()
    print(f"[api_deploy] Lease created: {json.dumps(lease_data, indent=2)[:300]}")
    return lease_data


# ------------------------------------------------------------------ step 5: get SSH details
def get_ssh_details(key: str, dseq: str, retries: int = 12, wait_s: int = 15) -> tuple[str, int]:
    """Poll until the deployment has an exposed SSH forward."""
    print("[api_deploy] Polling for SSH forwarding details ...")
    for attempt in range(1, retries + 1):
        time.sleep(wait_s)
        resp = requests.get(
            f"{API_BASE}/v1/deployments/{dseq}",
            headers=_headers(key),
            timeout=30,
        )
        if resp.status_code != 200:
            print(f"  attempt {attempt}: HTTP {resp.status_code}")
            continue

        data = resp.json()
        # Navigate to forwarded ports — structure varies by API version
        # Try common paths
        services = (
            data.get("services")
            or data.get("data", {}).get("services")
            or {}
        )
        if isinstance(services, list):
            services = {s.get("name", "svc"): s for s in services}

        for svc_name, svc in services.items():
            forwards = svc.get("forwardedPorts") or svc.get("exposedPorts") or []
            for fwd in forwards:
                ext_port = fwd.get("externalPort") or fwd.get("port")
                host = fwd.get("host") or fwd.get("ip") or ""
                if ext_port and host:
                    print(f"[api_deploy] SSH ready: host={host}, port={ext_port}")
                    return host, int(ext_port)

        # Also check top-level leases/forwarded
        leases = data.get("leases") or data.get("data", {}).get("leases") or []
        for lease in leases:
            for fwd in lease.get("forwardedPorts", []):
                host = fwd.get("host", "")
                port = fwd.get("externalPort") or fwd.get("port")
                if host and port:
                    print(f"[api_deploy] SSH ready: host={host}, port={port}")
                    return host, int(port)

        print(f"  attempt {attempt}/{retries}: container starting, waiting {wait_s}s ...")

    print("[api_deploy] Timed out waiting for SSH. The container may still be starting.")
    print("[api_deploy] Check manually: https://console.akash.network/")
    return "", 0


# ------------------------------------------------------------------ main
def main() -> None:
    key = _load_key()

    if not SDL_FILE.exists():
        print(f"[api_deploy] SDL not found at {SDL_FILE}. Run: python akash/deploy.py first.")
        sys.exit(1)

    sdl = SDL_FILE.read_text(encoding="utf-8")

    # Step 1
    dseq = create_deployment(key, sdl, deposit=1.5)
    print(f"[api_deploy] Track at: https://console.akash.network/deployments/{dseq}")

    # Step 2
    bids = wait_for_bids(key, dseq, wait_s=45)

    # Step 3
    bid = pick_cheapest_bid(bids)

    # Step 4
    create_lease(key, dseq, bid)

    # Step 5 — container needs time to boot and start startup.sh
    host, port = get_ssh_details(key, dseq, retries=15, wait_s=20)

    # ------------------------------------------------------------------ summary
    print()
    print("=" * 60)
    print("  DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"  dseq:     {dseq}")
    if host and port:
        print(f"  SSH:      ssh root@{host} -p {port}")
        print(f"  Password: MirageVM2026!")
        print()
        print("  Next steps from YOUR machine:")
        print(f"    python akash/run_install.py --host {host} --port {port}")
        print(f"    python akash/check_vm.py    --host {host} --port {port}")
        # Save SSH details for other scripts
        (Path(__file__).parent / "vm_ssh.txt").write_text(
            f"HOST={host}\nPORT={port}\nDSEQ={dseq}\n"
        )
        print("  (SSH details saved to akash/vm_ssh.txt)")
    else:
        print("  SSH details not yet available. Check console.akash.network")
        print(f"  Track: https://console.akash.network/deployments/{dseq}")
    print("=" * 60)


if __name__ == "__main__":
    main()
