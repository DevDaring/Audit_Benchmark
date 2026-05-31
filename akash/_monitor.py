"""
Local monitoring script — polls /data/logs on the running Akash VM
and reports pipeline phase, watchdog resource snapshots, and PIPELINE_COMPLETE.

Usage (after _deploy_mirage.py has run):
    python akash/_monitor.py

Or with explicit host/port:
    python akash/_monitor.py --host <IP> --port <PORT>

Reads akash/vm_ssh.txt for host/port if not provided.
"""
import argparse, sys, time, pathlib, re
try:
    import paramiko
except ImportError:
    print("pip install paramiko")
    sys.exit(1)

PASSWORD = "MirageVM2026!"

def read_vm_ssh():
    f = pathlib.Path("akash/vm_ssh.txt")
    if not f.exists():
        return None, 0
    host, port = "", 0
    for line in f.read_text().splitlines():
        if line.startswith("HOST=") and line[5:]:
            host = line[5:]
        if line.startswith("PORT=") and line[5:]:
            port = int(line[5:])
    return host, port


def connect(host, port, retries=5):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    for i in range(retries):
        try:
            client.connect(host, port=port, username="root",
                           password=PASSWORD, timeout=20, banner_timeout=30)
            return client
        except Exception as e:
            if i < retries - 1:
                print(f"  SSH connect failed ({e}), retry in 10s...")
                time.sleep(10)
    return None


def run_cmd(client, cmd, timeout=20):
    try:
        _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        return stdout.read().decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"[error: {e}]"


def poll_once(client):
    lines = []

    # Overall state markers
    state = run_cmd(client, "ls /data/state/ 2>/dev/null || echo '(no state dir)'")
    lines.append(f"  State markers: {state}")

    # Supervisor log tail
    sup = run_cmd(client, "tail -5 /data/logs/supervise.log 2>/dev/null || echo '(no supervise log)'")
    lines.append(f"  Supervisor:\n    " + "\n    ".join(sup.splitlines()))

    # Latest pipeline attempt log
    latest_log = run_cmd(client,
        "ls -t /data/logs/pipeline_attempt_*.log 2>/dev/null | head -1 || echo ''")
    if latest_log and latest_log != "''":
        tail = run_cmd(client, f"tail -10 '{latest_log}'")
        name = pathlib.Path(latest_log).name
        lines.append(f"  Pipeline ({name}):\n    " + "\n    ".join(tail.splitlines()))

    # Watchdog last line (resource snapshot)
    wdog = run_cmd(client, "tail -1 /data/logs/watchdog.log 2>/dev/null || echo '(watchdog not started)'")
    lines.append(f"  Watchdog: {wdog}")

    # Disk usage on persistent volume and ephemeral root
    disk = run_cmd(client, "df -h / /data 2>/dev/null | tail -3")
    lines.append(f"  Disk:\n    " + "\n    ".join(disk.splitlines()))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Monitor MIRAGE Akash VM pipeline")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    args = parser.parse_args()

    host, port = args.host, args.port
    if not host or not port:
        host, port = read_vm_ssh()
    if not host or not port:
        print("No host/port found. Run _deploy_mirage.py first, or pass --host --port")
        sys.exit(1)

    print(f"[monitor] Connecting to ssh root@{host} -p {port}")
    print(f"[monitor] Polling every {args.interval}s. Ctrl-C to stop.\n")

    poll = 0
    while True:
        poll += 1
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"{'='*60}")
        print(f"[{ts}] Poll #{poll}")

        client = connect(host, port, retries=3)
        if not client:
            print("  Could not connect — VM may be restarting (expected during evictions)")
        else:
            status = poll_once(client)
            client.close()
            print(status)

            # Check for completion
            if "PIPELINE_COMPLETE" in status:
                print("\n" + "="*60)
                print("  PIPELINE_COMPLETE — dry run passed!")
                print("  Retrieve results:")
                print(f"    ssh root@{host} -p {port}")
                print("    tail /data/logs/pipeline_attempt_*.log")
                print("    tail -50 /data/logs/watchdog.log  (root cause diagnostics)")
                print("="*60)
                break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
