import requests, json, time

key = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        key = line.split('=',1)[1].strip()

headers = {'x-api-key': key, 'Content-Type': 'application/json'}
BASE = 'https://console-api.akash.network'

# Check current deployments
r = requests.get(BASE + '/v1/deployments', headers=headers, timeout=30)
print("GET /v1/deployments:", r.status_code)
deps = r.json()
if isinstance(deps, dict):
    deps = deps.get('deployments') or deps.get('data') or [deps]
elif isinstance(deps, list):
    pass
print("Active deployments:", len(deps) if isinstance(deps, list) else type(deps))
if isinstance(deps, list):
    for d in deps[:5]:
        print(" dseq=%s status=%s" % (d.get('dseq'), d.get('status')))

# Use the latest dseq
dseq = '27070449'
print()
print("Checking bids for dseq=%s" % dseq)

# Check deployment status
r = requests.get(BASE + '/v1/deployments/' + dseq, headers=headers, timeout=30)
print("GET /v1/deployments/%s: %s" % (dseq, r.status_code))
print(json.dumps(r.json(), indent=2)[:2000])

print()
# Check bids
r = requests.get(BASE + '/v1/bids', headers=headers, params={'dseq': dseq}, timeout=30)
print("GET /v1/bids?dseq=%s: %s" % (dseq, r.status_code))
print(json.dumps(r.json(), indent=2)[:2000])
