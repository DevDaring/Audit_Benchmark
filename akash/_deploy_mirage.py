"""
MIRAGE Akash deployment — full correct API flow.

POST /v1/deployments  -> extracts dseq + manifest blob
GET  /v1/bids         -> wait for bid
POST /v1/leases       -> { "manifest": <blob>, "leases": [{dseq, gseq, oseq, provider}] }
Poll /v1/deployments/{dseq} -> get forwarded SSH port
Upload .env via SFTP
"""

import json, requests, time, sys
from pathlib import Path

KEY = ''
for line in open('Code/mirage/.env').read().splitlines():
    if line.startswith('AKASH_API_KEY='):
        KEY = line.split('=',1)[1].strip()

H = {'x-api-key': KEY, 'Content-Type': 'application/json'}
BASE = 'https://console-api.akash.network'
ENV_FILE = Path('Code/mirage/.env')

SDL_MIRAGE = """\
---
version: "2.0"

services:
  mirage:
    image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
    env:
      - GITHUB_REPO=https://github.com/DevDaring/Audit_Benchmark.git
      - ROOT_PASSWORD=MirageVM2026!
    command:
      - bash
      - -c
      - |
        apt-get update -qq && apt-get install -y git curl tmux openssh-server wget > /dev/null
        mkdir -p /workspace
        echo "root:MirageVM2026!" | chpasswd
        mkdir -p /run/sshd
        sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
        sed -i "s/#PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
        /usr/sbin/sshd
        cd /workspace
        git clone $GITHUB_REPO Audit_Benchmark || git -C Audit_Benchmark pull
        echo "VM_READY" > /workspace/vm_ready.txt
        tail -f /dev/null
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
          units: 4
        memory:
          size: 64Gi
        storage:
          - size: 200Gi
        gpu:
          units: 1
          attributes:
            vendor:
              nvidia:

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

# ── close old open deployments ────────────────────────────────────────────
print("[deploy] Closing any leftover open deployments ...")
for old in ['27070590', '27070564', '27070538', '27070733']:
    requests.delete(BASE + '/v1/deployments/' + old, headers=H, timeout=20)

# ── 1. create deployment ──────────────────────────────────────────────────
print("[deploy] Creating deployment ...")
resp = requests.post(BASE + '/v1/deployments', headers=H,
    json={'data': {'sdl': SDL_MIRAGE, 'deposit': 2.0}}, timeout=60)

if resp.status_code not in (200, 201):
    print("FAILED %d: %s" % (resp.status_code, resp.text[:600]))
    sys.exit(1)

full_resp = resp.json()
print("Response keys:", list(full_resp.get('data', full_resp).keys()) if isinstance(full_resp, dict) else "list")

data = full_resp.get('data', full_resp) if isinstance(full_resp, dict) else full_resp
dseq = str(data.get('dseq', ''))
manifest_blob = data.get('manifest', '')  # <<< critical: from deployment response

print("[deploy] dseq=%s  manifest_len=%d" % (dseq, len(str(manifest_blob))))
if not manifest_blob:
    print("[deploy] WARNING: manifest not in response. Full response:")
    print(json.dumps(full_resp, indent=2)[:2000])

# ── 2. wait for bid ───────────────────────────────────────────────────────
print("[deploy] Waiting for bid (up to 90s) ...")
bid = None
for i in range(1, 10):
    time.sleep(10)
    r = requests.get(BASE + '/v1/bids', headers=H, params={'dseq': dseq}, timeout=30)
    bids = r.json()
    bids_list = bids.get('data', bids) if isinstance(bids, dict) else bids
    n = len(bids_list) if isinstance(bids_list, list) else 0
    print("  t=%ds: %d bid(s)" % (i*10, n))
    if n > 0:
        bid = bids_list[0]
        break

if not bid:
    print("[deploy] No bids. Closing.")
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
        gpu_attrs.append(attr.get('key',''))

print("[deploy] Bid from: %s | price: %s %s/block | GPU: %s" % (
    provider, price.get('amount','?'), price.get('denom','?'), gpu_attrs))

# ── 3. create lease (correct API format) ──────────────────────────────────
print("[deploy] Creating lease ...")
lease_payload = {
    "manifest": manifest_blob,
    "leases": [{"dseq": dseq, "gseq": gseq, "oseq": oseq, "provider": provider}]
}
lr = requests.post(BASE + '/v1/leases', headers=H, json=lease_payload, timeout=60)
print("[deploy] Lease: %d" % lr.status_code)
if lr.status_code not in (200, 201):
    print("Lease response:", lr.text[:600])
    # Continue anyway — check if lease exists on-chain
else:
    print("Lease created successfully!")

# ── 4. poll for SSH ───────────────────────────────────────────────────────
print("[deploy] Polling for SSH details (up to 8 min) ...")
host, port = '', 0
for j in range(1, 33):
    time.sleep(15)
    dr = requests.get(BASE + '/v1/deployments/' + dseq, headers=H, timeout=30)
    if dr.status_code != 200:
        continue
    dd = dr.json().get('data', {})

    # Recursive search for forwarded ports
    def find_port(obj, depth=0):
        if depth > 8 or not obj:
            return None, None
        if isinstance(obj, dict):
            h = obj.get('host') or obj.get('ip') or obj.get('externalIp') or ''
            p = obj.get('externalPort') or obj.get('port')
            if h and p and str(p) not in ('0', '22'):
                return h, int(p)
            if h and p and str(p) == '22':
                return h, int(p)
            for v in obj.values():
                res = find_port(v, depth+1)
                if res[0]:
                    return res
        elif isinstance(obj, list):
            for item in obj:
                res = find_port(item, depth+1)
                if res[0]:
                    return res
        return None, None

    fh, fp = find_port(dd)
    if fh and fp:
        host, port = fh, fp
        print("\n[deploy] SSH READY: ssh root@%s -p %s" % (host, port))
        break

    # Every 60s print a status line
    if j % 4 == 0:
        leases = dd.get('leases', [])
        print("  t=%ds: %d lease(s), container still starting ..." % (j*15, len(leases)))

if not host:
    print("[deploy] Could not auto-detect SSH.")
    print("[deploy] Check: https://console.akash.network/deployments/%s" % dseq)
    Path('akash/vm_ssh.txt').write_text("DSEQ=%s\nHOST=\nPORT=\n" % dseq)
    print("Once you have IP+port, run:  python akash/run_install.py --host <IP> --port <PORT>")
    sys.exit(0)

# ── 5. upload .env ────────────────────────────────────────────────────────
print("[deploy] Uploading .env ...")
try:
    import paramiko
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for attempt in range(1, 8):
        try:
            client.connect(host, port=port, username='root',
                           password='MirageVM2026!', timeout=30, banner_timeout=60)
            break
        except Exception as e:
            print("  SSH attempt %d: %s" % (attempt, e))
            time.sleep(15)

    client.exec_command('mkdir -p /workspace/Audit_Benchmark/Code/mirage')
    time.sleep(2)
    sftp = client.open_sftp()
    sftp.put(str(ENV_FILE), '/workspace/Audit_Benchmark/Code/mirage/.env')
    sftp.close()
    client.exec_command('chmod 600 /workspace/Audit_Benchmark/Code/mirage/.env')
    client.close()
    print("[deploy] .env uploaded!")
except Exception as exc:
    print("[deploy] .env upload: %s" % exc)
    print("  Upload manually:  scp -P %s Code/mirage/.env root@%s:/workspace/Audit_Benchmark/Code/mirage/.env" % (port, host))

# ── save + summary ────────────────────────────────────────────────────────
Path('akash/vm_ssh.txt').write_text("HOST=%s\nPORT=%s\nDSEQ=%s\n" % (host, port, dseq))
print()
print("=" * 70)
print("  MIRAGE VM READY")
print("=" * 70)
print("  dseq:     %s" % dseq)
print("  SSH:      ssh root@%s -p %s" % (host, port))
print("  Password: MirageVM2026!")
print("  GPU:      %s" % gpu_attrs)
print()
print("  NEXT STEPS (run from your machine):")
print()
print("  1. Install all packages + dry run:")
print("     python akash/run_install.py --host %s --port %s" % (host, port))
print()
print("  2. Health check:")
print("     python akash/check_vm.py --host %s --port %s" % (host, port))
print()
print("  3. GPU pipeline:")
print("     python akash/run_install.py --host %s --port %s --action gpu_run" % (host, port))
print("=" * 70)
