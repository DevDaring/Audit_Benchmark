import json, requests

key = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        key = line.split('=',1)[1].strip()

headers = {'x-api-key': key}
resp = requests.get('https://console-api.akash.network/v1/providers', headers=headers, params={'gpu': 'true'}, timeout=30)
providers = resp.json()

print("Total providers:", len(providers))
print()

gpu_providers = []
for p in providers:
    gpu_stats = p.get('stats', {}).get('gpu', {})
    gpu_avail = gpu_stats.get('available', 0)
    gpu_total = gpu_stats.get('total', 0)
    gpu_models = p.get('gpuModels') or p.get('hardwareGpuModels') or []
    is_online = p.get('isOnline', False)
    if gpu_total > 0 or gpu_avail > 0:
        gpu_providers.append({
            'owner': p.get('owner','')[:24],
            'hostUri': p.get('hostUri',''),
            'online': is_online,
            'gpu_avail': gpu_avail,
            'gpu_total': gpu_total,
            'models': gpu_models,
        })

print("Providers with GPU (total>0):", len(gpu_providers))
for gp in sorted(gpu_providers, key=lambda x: -x['gpu_avail']):
    models_str = str(gp['models'])[:60]
    print("  online=%s avail=%s/%s models=%s" % (gp['online'], gp['gpu_avail'], gp['gpu_total'], models_str))
    print("  host=%s" % gp['hostUri'][:80])

# Also show online providers
print()
online_count = sum(1 for p in providers if p.get('isOnline'))
print("Online providers:", online_count)
