"""
Minimal test deployment requesting ANY 1 nvidia GPU to confirm bid flow works.
"""
import json, requests, time, sys

key = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        key = line.split('=',1)[1].strip()

headers = {'x-api-key': key, 'Content-Type': 'application/json'}
BASE = 'https://console-api.akash.network'

# MINIMAL SDL: any 1 nvidia GPU, ubuntu, just sleep
SDL_MINIMAL = """\
---
version: "2.0"

services:
  test:
    image: ubuntu:22.04
    command:
      - bash
      - -c
      - sleep 300
    expose:
      - port: 8080
        as: 8080
        to:
          - global: false

profiles:
  compute:
    test:
      resources:
        cpu:
          units: 4
        memory:
          size: 16Gi
        storage:
          - size: 20Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:

  placement:
    akash:
      pricing:
        test:
          denom: uakt
          amount: 10000000

deployment:
  test:
    akash:
      profile: test
      count: 1
"""

print("[test] Creating minimal ANY-GPU deployment ...")
resp = requests.post(BASE + '/v1/deployments', headers=headers,
    json={'data': {'sdl': SDL_MINIMAL, 'deposit': 2.0}}, timeout=60)
print("Create: %d" % resp.status_code)
if resp.status_code not in (200, 201):
    print(resp.text[:500])
    sys.exit(1)

data = resp.json()
dseq = str(data.get('dseq') or data.get('data', {}).get('dseq', ''))
print("dseq=%s" % dseq)

# Wait for bids
for i in range(1, 13):
    time.sleep(10)
    r = requests.get(BASE + '/v1/bids', headers=headers, params={'dseq': dseq}, timeout=30)
    bids = r.json().get('data', r.json()) if isinstance(r.json(), dict) else r.json()
    n = len(bids) if isinstance(bids, list) else 0
    print("  t=%ds: %d bid(s)" % (i*10, n))
    if n > 0:
        print("BIDS RECEIVED:")
        print(json.dumps(bids, indent=2)[:2000])
        # Close this test deployment
        requests.delete(BASE + '/v1/deployments/' + dseq, headers=headers, timeout=30)
        print("Test deployment closed.")
        print("CONCLUSION: Akash bid flow works. Issue was GPU model constraint.")
        sys.exit(0)

print("No bids even for ANY-GPU. Closing test deployment.")
requests.delete(BASE + '/v1/deployments/' + dseq, headers=headers, timeout=30)
print("CONCLUSION: API issue or all providers offline. Check console.akash.network")
