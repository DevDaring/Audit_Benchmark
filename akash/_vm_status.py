"""Quick VM status check."""
import paramiko, sys

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 32355

def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
              timeout=12, banner_timeout=20)
    return c

def run(c, cmd):
    _, o, e = c.exec_command(cmd, timeout=10)
    return (o.read() + e.read()).decode(errors="replace").strip()

try:
    c = conn()
    print("=== VM STATUS ===")
    print(f"Container age (PID1 seconds): {run(c, 'ps -p 1 -o etimes=')}")
    print(f"Host load avg:  {run(c, 'cat /proc/loadavg')}")
    print(f"Cgroup mem limit: {int(run(c, 'cat /sys/fs/cgroup/memory.max')) // 1073741824} GiB")
    print(f"Cgroup mem used:  {int(run(c, 'cat /sys/fs/cgroup/memory.current')) // 1048576} MB")
    print(f"GPU mem used/free: {run(c, 'nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo n/a')}")
    print(f"Workspace: {run(c, 'ls /workspace/')}")
    print(f"Tmux sessions: {run(c, 'tmux list-sessions 2>/dev/null || echo none')}")
    print(f"Pipeline log: {run(c, 'wc -c /workspace/full_pipeline.log 2>/dev/null || echo missing')}")
    print(f"Log tail:\n{run(c, 'tail -10 /workspace/full_pipeline.log 2>/dev/null || echo (no log)')}")
    c.close()
except Exception as ex:
    print(f"Cannot connect: {ex}")
    sys.exit(1)
