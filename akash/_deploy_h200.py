"""
Fallback: deploy on H200 141GB instead of A100 80GB.
14 units available at provider.h200.atl.val.akash.pub

H200 is Hopper architecture (SM90) — supports Flash Attention 3 and is
fully backward-compatible with Flash Attention 2.  VRAM budget:
  Behavioral per model: up to 28 GB (Phi-4 14B) -> fits trivially in 141 GB
  CDVA (HF + TL): up to 36 GB -> fits trivially in 141 GB
"""

import json, requests, time, sys
from pathlib import Path

key = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        key = line.split('=',1)[1].strip()

headers = {'x-api-key': key, 'Content-Type': 'application/json'}
BASE = 'https://console-api.akash.network'
ENV_B64_LINE = ''
for line in open('akash/deployment.yaml').read().splitlines():
    if 'MIRAGE_ENV_B64=' in line:
        ENV_B64_LINE = line.strip()
        break

SDL_H200 = """---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
      - {env_b64_line}
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
                - model: h200

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
""".format(env_b64_line=ENV_B64_LINE)

# Create deployment
print("[h200_deploy] Creating H200 deployment ...")
resp = requests.post(BASE + '/v1/deployments', headers=headers,
    json={'data': {'sdl': SDL_H200, 'deposit': 2.0}}, timeout=60)

if resp.status_code not in (200, 201):
    print("FAILED:", resp.status_code, resp.text[:500])
    sys.exit(1)

data = resp.json()
dseq = str(data.get('dseq') or data.get('data', {}).get('dseq', ''))
print("[h200_deploy] dseq=%s" % dseq)
print("[h200_deploy] Track: https://console.akash.network/deployments/%s" % dseq)

# Wait for bids
print("[h200_deploy] Waiting 90s for bids ...")
for i in range(1, 10):
    time.sleep(10)
    r = requests.get(BASE + '/v1/bids', headers=headers, params={'dseq': dseq}, timeout=30)
    bids = r.json()
    data = bids.get('data', bids) if isinstance(bids, dict) else bids
    n = len(data) if isinstance(data, list) else 0
    print("  t=%ds: %d bid(s)" % (i*10, n))
    if n > 0:
        print("BIDS:")
        print(json.dumps(data, indent=2)[:2000])
        # Accept cheapest
        def price(b):
            p = b.get('price') or b.get('amount') or {}
            return float(p.get('amount', 999999) if isinstance(p, dict) else p or 999999)
        chosen = sorted(data, key=price)[0]
        provider = chosen.get('provider') or chosen.get('providerId')
        gseq = chosen.get('gseq', 1)
        oseq = chosen.get('oseq', 1)
        print("\n[h200_deploy] Accepting bid from %s ..." % provider)
        lr = requests.post(BASE + '/v1/leases', headers=headers,
            json={'data': [{'dseq': dseq, 'gseq': gseq, 'oseq': oseq, 'provider': provider}]},
            timeout=60)
        print("Lease: %d %s" % (lr.status_code, lr.text[:300]))
        
        # Wait for SSH
        print("[h200_deploy] Polling for SSH (up to 5 min) ...")
        for j in range(1, 19):
            time.sleep(15)
            dr = requests.get(BASE + '/v1/deployments/' + dseq, headers=headers, timeout=30)
            dd = dr.json().get('data', {})
            services = dd.get('services') or {}
            if isinstance(services, list):
                services = {s.get('name','s'): s for s in services}
            for sn, sv in services.items():
                for fwd in sv.get('forwardedPorts', []):
                    h = fwd.get('host') or fwd.get('ip', '')
                    p = fwd.get('externalPort') or fwd.get('port')
                    if h and p:
                        print("\nSSH READY: ssh root@%s -p %s" % (h, p))
                        Path('akash/vm_ssh.txt').write_text("HOST=%s\nPORT=%s\nDSEQ=%s\n" % (h, p, dseq))
                        sys.exit(0)
            print("  t=%ds: container starting ..." % (j*15))
        print("[h200_deploy] Container starting. Check console.akash.network")
        sys.exit(0)

print("[h200_deploy] No bids for H200 either. Trying without GPU model constraint ...")
# Try without any GPU model constraint
sys.exit(1)
