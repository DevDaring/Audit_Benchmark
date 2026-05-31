"""Check current deployment status and get SSH details."""
import json, requests, time

KEY = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        KEY = line.split('=',1)[1].strip()

H = {'x-api-key': KEY}
BASE = 'https://console-api.akash.network'
DSEQ = '27070590'

# Check deployment
r = requests.get(BASE + '/v1/deployments/' + DSEQ, headers=H, timeout=30)
print("Deployment status: %d" % r.status_code)
data = r.json()
print(json.dumps(data, indent=2)[:4000])

# Check bids
r2 = requests.get(BASE + '/v1/bids', headers=H, params={'dseq': DSEQ}, timeout=30)
print("\nBids: %d" % r2.status_code)
print(json.dumps(r2.json(), indent=2)[:2000])
