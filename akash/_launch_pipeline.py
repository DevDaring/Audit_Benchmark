"""
Launch dry run then GPU pipeline on the VM.
Steps:
  1. Run dry_run_all.py (dataset + cpu_only dry runs)
  2. Run dry_run_gpu_cpu.py (GPU smoke test)
  3. Launch full pipeline in tmux: osm_behavioral -> cdva_patching -> calibration -> leaderboard
"""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594
REMOTE_CODE = '/workspace/Audit_Benchmark/Code/mirage'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=300):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = (o.read() + e.read()).decode('utf-8', 'replace').strip()
    rc = o.channel.recv_exit_status()
    return rc, out

def run_stream(cmd, timeout=300):
    """Run command and stream output."""
    transport = c.get_transport()
    chan = transport.open_session()
    chan.set_combine_stderr(True)
    chan.exec_command(cmd)
    while True:
        chunk = chan.recv(4096)
        if not chunk:
            break
        sys.stdout.write(chunk.decode('utf-8', 'replace'))
        sys.stdout.flush()
    return chan.recv_exit_status()

print("=" * 60)
print("STEP 1: Setting up HF token and paths")
print("=" * 60)
# Extract HF token from .env
rc, hf_token = run("grep '^HUGGINGFACE_TOKEN=' %s/.env | cut -d= -f2" % REMOTE_CODE)
print("HF_TOKEN found:", bool(hf_token.strip()))

# Set up environment
setup_cmd = """
export HF_HOME=/workspace/.hf_cache
export TRANSFORMERS_CACHE=/workspace/.hf_cache/hub
mkdir -p /workspace/.hf_cache
mkdir -p %s/Results
""" % REMOTE_CODE
run(setup_cmd)

print()
print("=" * 60)
print("STEP 2: dry_run_dataset.py")
print("=" * 60)
rc = run_stream("cd %s && python3 Dry_Run/dry_run_dataset.py 2>&1" % REMOTE_CODE, timeout=180)
print("\ndry_run_dataset exit=%d" % rc)

print()
print("=" * 60)
print("STEP 3: dry_run_cpu_only.py")
print("=" * 60)
rc = run_stream("cd %s && python3 Dry_Run/dry_run_cpu_only.py 2>&1" % REMOTE_CODE, timeout=180)
print("\ndry_run_cpu_only exit=%d" % rc)

print()
print("=" * 60)
print("STEP 4: dry_run_gpu_cpu.py (GPU smoke test)")
print("=" * 60)
rc = run_stream(
    "cd %s && HF_HOME=/workspace/.hf_cache python3 Dry_Run/dry_run_gpu_cpu.py 2>&1" % REMOTE_CODE,
    timeout=300
)
print("\ndry_run_gpu_cpu exit=%d" % rc)

print()
print("=" * 60)
print("STEP 5: Launch full GPU pipeline in tmux 'gpu_run'")
print("=" * 60)

PIPELINE_CMD = (
    "cd %s && "
    "export HF_HOME=/workspace/.hf_cache && "
    "export TRANSFORMERS_CACHE=/workspace/.hf_cache/hub && "
    "mkdir -p /workspace/logs && "
    "python3 GPU_CPU/osm_behavioral.py && "
    "python3 GPU_CPU/cdva_patching.py && "
    "python3 GPU_CPU/cdva_calibration.py && "
    "python3 CPU_Only/leaderboard.py && "
    "echo GPU_PIPELINE_DONE"
) % REMOTE_CODE

run("tmux kill-session -t gpu_run 2>/dev/null || true")
run("tmux new-session -d -s gpu_run -x 250 -y 50")
run("tmux send-keys -t gpu_run 'bash -c \"%s\" 2>&1 | tee /workspace/logs/gpu_run.log; echo EXIT_CODE=$?' Enter" % PIPELINE_CMD)

time.sleep(3)
rc2, ps = run("tmux ls 2>&1")
print("tmux sessions:", ps)

c.close()

print()
print("=" * 70)
print("  GPU PIPELINE LAUNCHED IN TMUX SESSION 'gpu_run'")
print("=" * 70)
print("  SSH:  ssh root@%s -p %s" % (HOST, PORT))
print("  Watch: tmux attach -t gpu_run")
print("  Log:   tail -f /workspace/logs/gpu_run.log")
print("=" * 70)
