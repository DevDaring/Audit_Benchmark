"""Wait up to 3 minutes for bids on the current active deployment."""
import requests, json, time

key = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        key = line.split('=',1)[1].strip()

headers = {'x-api-key': key}
BASE = 'https://console-api.akash.network'

dseq = '27070449'

for attempt in range(1, 19):
    time.sleep(10)
    r = requests.get(BASE + '/v1/bids', headers=headers, params={'dseq': dseq}, timeout=30)
    bids = r.json()
    data = bids.get('data', bids) if isinstance(bids, dict) else bids
    if isinstance(data, list) and len(data) > 0:
        print("BIDS FOUND after %ds:" % (attempt * 10))
        print(json.dumps(data, indent=2)[:3000])
        break
    print("  t=%ds: %d bids" % (attempt * 10, len(data) if isinstance(data, list) else 0))
else:
    print("No bids after 3 minutes. Deployment dseq=%s is still open." % dseq)
    print()
    # Also try different endpoints
    for ep in ['/v1/orders', '/v1/bids/' + dseq]:
        r2 = requests.get(BASE + ep, headers=headers, timeout=15)
        print("GET %s: %s - %s" % (ep, r2.status_code, r2.text[:200]))
