"""
MIRAGE Akash deployment — P1a SDL with persistent volume.

Fixes applied based on expert review (Akash_Restart_Issue.md):
  - Root cause was DISK eviction (ephemeral-storage limit), NOT memory OOM.
  - Persistent /data volume (120Gi, beta3/NVMe) for venv + HF cache + state.
  - Smaller ephemeral root (30Gi) — small footprint = last to be evicted.
  - All caches redirected to /data via SDL env vars (HF_HOME, PIP_CACHE_DIR…).
  - Supervisor (supervise_pipeline.sh) + watchdog auto-start on every boot.
  - Wider GPU list (24–80 GB) to reach calmer, less-contended providers.
  - Runtime Docker image (not devel) — smaller imagefs footprint (P2).

After deploy: python akash/_monitor.py  (polls /data/logs until PIPELINE_COMPLETE)
"""

import json, requests, time, sys
from pathlib import Path

KEY = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        KEY = line.split('=', 1)[1].strip().strip('"').strip("'")

H    = {'x-api-key': KEY, 'Content-Type': 'application/json'}
BASE = 'https://console-api.akash.network'
ENV_FILE = Path('Code/mirage/.env')

# ── New SDL — persistent volume + supervisor + wider GPU set ─────────────
SDL_MIRAGE = """\
---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
      - HF_HOME=/data/hf_cache
      - HUGGINGFACE_HUB_CACHE=/data/hf_cache/hub
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - PIP_CACHE_DIR=/data/pip_cache
      - XDG_CACHE_HOME=/data/cache
      - VENV=/data/venv
      - STATE_DIR=/data/state
      - REPO_DIR=/data/Audit_Benchmark
    command:
      - bash
      - -c
      - |
        apt-get update -qq && apt-get install -y git curl tmux openssh-server wget python3-venv python3-dev build-essential > /dev/null 2>&1
        rm -rf /var/lib/apt/lists/*
        echo "root:MirageVM2026!" | chpasswd
        mkdir -p /run/sshd
        sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
        sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
        /usr/sbin/sshd
        mkdir -p /data/logs /data/state /data/hf_cache /data/pip_cache /data/cache /workspace
        if [ ! -d /data/Audit_Benchmark/.git ]; then
          git clone https://github.com/DevDaring/Audit_Benchmark.git /data/Audit_Benchmark 2>&1 || true
        else
          git -C /data/Audit_Benchmark pull --ff-only 2>&1 || true
        fi
        echo "VM_READY $(date -u +%FT%TZ)" > /workspace/vm_ready.txt
        nohup bash /data/Audit_Benchmark/akash/watchdog.sh >> /data/logs/watchdog_boot.log 2>&1 &
        nohup bash /data/Audit_Benchmark/akash/supervise_pipeline.sh >> /data/logs/supervise.log 2>&1 &
        tail -f /dev/null
    expose:
      - port: 22
        as: 22
        to:
          - global: true
    params:
      storage:
        data:
          mount: /data
          readOnly: false

profiles:
  compute:
    mirage:
      resources:
        cpu:
          units: 4
        memory:
          size: 64Gi
        storage:
          - size: 30Gi
          - name: data
            size: 120Gi
            attributes:
              persistent: true
              class: beta3
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:
                - model: rtx4090
                - model: rtx3090
                - model: a10
                - model: l4
                - model: a40
                - model: a6000
                - model: l40
                - model: l40s
                - model: a100

  placement:
    akash:
      pricing:
        mirage:
          denom: uakt
          amount: 10000000

deployment:
  mirage:
    akash:
      profile: mirage
      count: 1
"""

# ── Close previous deployments ────────────────────────────────────────────
print("[deploy] Closing any leftover open deployments ...")
OLD_DSEQS = ['27070590', '27070564', '27070538', '27070733', '27071620']
for old in OLD_DSEQS:
    r = requests.delete(BASE + '/v1/deployments/' + old, headers=H, timeout=20)
    if r.status_code in (200, 201):
        print(f"  Closed DSEQ {old}")

# ── 1. Create deployment ──────────────────────────────────────────────────
print("[deploy] Creating deployment with persistent volume SDL ...")
resp = requests.post(BASE + '/v1/deployments', headers=H,
    json={'data': {'sdl': SDL_MIRAGE, 'deposit': 5.0}}, timeout=60)

if resp.status_code not in (200, 201):
    print("FAILED %d: %s" % (resp.status_code, resp.text[:600]))
    sys.exit(1)

full_resp = resp.json()
data = full_resp.get('data', full_resp) if isinstance(full_resp, dict) else full_resp
dseq = str(data.get('dseq', ''))
manifest_blob = data.get('manifest', '')

print("[deploy] dseq=%s  manifest_len=%d" % (dseq, len(str(manifest_blob))))
if not manifest_blob:
    print("[deploy] WARNING: manifest not in response. Full response:")
    print(json.dumps(full_resp, indent=2)[:2000])

# ── 2. Wait for bid ───────────────────────────────────────────────────────
print("[deploy] Waiting for bid (up to 120s — beta3 providers may take longer) ...")
bid = None
for i in range(1, 13):
    time.sleep(10)
    r = requests.get(BASE + '/v1/bids', headers=H, params={'dseq': dseq}, timeout=30)
    bids = r.json()
    bids_list = bids.get('data', bids) if isinstance(bids, dict) else bids
    n = len(bids_list) if isinstance(bids_list, list) else 0
    print("  t=%ds: %d bid(s)" % (i * 10, n))
    if n > 0:
        bid = bids_list[0]
        break

if not bid:
    print("[deploy] No bids with beta3 storage. Retrying with beta2 (SSD) ...")
    # Fallback: replace beta3 with beta2 and try again
    sdl_beta2 = SDL_MIRAGE.replace("class: beta3", "class: beta2")
    resp2 = requests.post(BASE + '/v1/deployments', headers=H,
        json={'data': {'sdl': sdl_beta2, 'deposit': 5.0}}, timeout=60)
    if resp2.status_code in (200, 201):
        data2 = resp2.json().get('data', resp2.json())
        dseq = str(data2.get('dseq', dseq))
        manifest_blob = data2.get('manifest', manifest_blob)
        print("[deploy] beta2 deployment dseq=%s — waiting for bids..." % dseq)
        for i in range(1, 10):
            time.sleep(10)
            r = requests.get(BASE + '/v1/bids', headers=H, params={'dseq': dseq}, timeout=30)
            bids = r.json()
            bids_list = bids.get('data', bids) if isinstance(bids, dict) else bids
            n = len(bids_list) if isinstance(bids_list, list) else 0
            print("  beta2 t=%ds: %d bid(s)" % (i * 10, n))
            if n > 0:
                bid = bids_list[0]
                break

if not bid:
    print("[deploy] No bids after beta3 + beta2. Closing.")
    requests.delete(BASE + '/v1/deployments/' + dseq, headers=H, timeout=20)
    sys.exit(1)

bid_data = bid.get('bid', bid)
bid_id = bid_data.get('id', {})
provider = bid_id.get('provider') or bid.get('provider')
gseq = bid_id.get('gseq', 1)
oseq = bid_id.get('oseq', 1)
price = bid_data.get('price', {})
gpu_attrs = []
for res in (bid_data.get('resources_offer') or []):
    for attr in res.get('resources', {}).get('gpu', {}).get('attributes', []):
        gpu_attrs.append(attr.get('key', '') + '=' + attr.get('value', ''))

print("[deploy] Bid from: %s | price: %s %s/block | GPU: %s" % (
    provider, price.get('amount', '?'), price.get('denom', '?'), gpu_attrs))

# ── 3. Create lease ───────────────────────────────────────────────────────
print("[deploy] Creating lease ...")
lr = requests.post(BASE + '/v1/leases', headers=H, json={
    "manifest": manifest_blob,
    "leases": [{"dseq": dseq, "gseq": gseq, "oseq": oseq, "provider": provider}]
}, timeout=60)
print("[deploy] Lease: %d" % lr.status_code)
if lr.status_code not in (200, 201):
    print("Lease response:", lr.text[:600])
else:
    print("Lease created successfully!")

# ── 4. Poll for SSH ───────────────────────────────────────────────────────
print("[deploy] Polling for SSH details (up to 10 min — persistent volume mount adds ~2 min) ...")
host, port = '', 0
for j in range(1, 41):
    time.sleep(15)
    dr = requests.get(BASE + '/v1/deployments/' + dseq, headers=H, timeout=30)
    if dr.status_code != 200:
        continue
    dd = dr.json().get('data', {})

    def find_port(obj, depth=0):
        if depth > 8 or not obj:
            return None, None
        if isinstance(obj, dict):
            h = obj.get('host') or obj.get('ip') or obj.get('externalIp') or ''
            p = obj.get('externalPort') or obj.get('port')
            if h and p:
                return h, int(p)
            for v in obj.values():
                res = find_port(v, depth + 1)
                if res[0]:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_port(item, depth + 1)
                if res[0]:
                    return res
        return None, None

    fh, fp = find_port(dd)
    if fh and fp:
        host, port = fh, fp
        print("\n[deploy] SSH READY: ssh root@%s -p %s" % (host, port))
        break
    if j % 4 == 0:
        print("  t=%ds: container still starting ..." % (j * 15))

if not host:
    print("[deploy] Could not auto-detect SSH. Check Akash Console.")
    print("[deploy] https://console.akash.network/deployments/%s" % dseq)
    Path('akash/vm_ssh.txt').write_text("DSEQ=%s\nHOST=\nPORT=\n" % dseq)
    sys.exit(0)

# ── 5. Upload .env to /data/.env (persistent — survives all restarts) ────
print("[deploy] Uploading .env to /data/.env (persistent volume) ...")
try:
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, 10):
        try:
            client.connect(host, port=port, username='root',
                           password='MirageVM2026!', timeout=30, banner_timeout=60)
            break
        except Exception as e:
            print("  SSH attempt %d: %s" % (attempt, e))
            time.sleep(15)

    # Ensure /data is mounted (may take a moment on first boot)
    client.exec_command('mkdir -p /data && ls /data', timeout=10)
    time.sleep(3)

    sftp = client.open_sftp()
    sftp.put(str(ENV_FILE), '/data/.env')
    sftp.close()
    client.exec_command('chmod 600 /data/.env', timeout=5)
    client.close()
    print("[deploy] .env uploaded to /data/.env (persistent — will not need re-upload)")
except Exception as exc:
    print("[deploy] .env upload failed: %s" % exc)
    print("  Upload manually:  scp -P %s Code/mirage/.env root@%s:/data/.env" % (port, host))

# ── Save + summary ────────────────────────────────────────────────────────
Path('akash/vm_ssh.txt').write_text(
    "HOST=%s\nPORT=%s\nDSEQ=%s\n" % (host, port, dseq))

print()
print("=" * 70)
print("  MIRAGE VM READY (persistent volume deployment)")
print("=" * 70)
print("  dseq:       %s" % dseq)
print("  SSH:        ssh root@%s -p %s" % (host, port))
print("  Password:   MirageVM2026!")
print("  GPU:        %s" % gpu_attrs)
print("  .env:       /data/.env (persistent — survives restarts)")
print()
print("  The supervisor + watchdog are already running automatically.")
print("  Monitor progress:")
print("    python akash/_monitor.py")
print()
print("  After PIPELINE_COMPLETE, check watchdog for eviction root cause:")
print("    tail -50 /data/logs/watchdog.log")
print()
print("  To close the deployment (stop billing):")
print("    DELETE https://console-api.akash.network/v1/deployments/%s" % dseq)
print("=" * 70)
