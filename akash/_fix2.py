"""Fix missing packages using python3 -m pip, then run dry runs and GPU pipeline."""
import paramiko, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HOST = 'provider.a100.dsm.val.akash.pub'
PORT = 30594
CODE = '/workspace/Audit_Benchmark/Code/mirage'

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username='root', password='MirageVM2026!', timeout=30)

def run(cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = (o.read() + e.read()).decode('utf-8', 'replace').strip()
    rc = o.channel.recv_exit_status()
    return rc, out

def stream(cmd, timeout=300):
    t = c.get_transport()
    ch = t.open_session()
    ch.set_combine_stderr(True)
    ch.exec_command(cmd)
    while True:
        chunk = ch.recv(4096)
        if not chunk:
            break
        sys.stdout.write(chunk.decode('utf-8', 'replace'))
        sys.stdout.flush()
    return ch.recv_exit_status()

# Install ALL missing packages using python3 -m pip
print("=== Installing missing packages (python3 -m pip) ===")
rc, out = run(
    "python3 -m pip install "
    "python-dotenv>=1.0.0 "
    "outlines>=0.1.0 "
    "boto3>=1.34.0 "
    "google-generativeai>=0.8.0 "
    "mistralai>=1.0.0 "
    "openai>=1.35.0 "
    "scipy>=1.11.0 "
    "paramiko>=3.4.0 "
    "2>&1 | tail -8",
    timeout=120
)
print(out)
print("pip exit=%d" % rc)

# Verify critical imports
print()
print("=== Verifying imports ===")
_, v = run("python3 -c 'import dotenv, boto3, openai; print(\"dotenv+boto3+openai OK\")'")
print(v)
_, v2 = run("python3 -c 'import outlines; print(\"outlines OK\")'")
print(v2)
_, v3 = run("python3 -c 'from dotenv import load_dotenv; print(\"load_dotenv OK\")'")
print(v3)

# Kill old bad sessions
run("tmux kill-session -t gpu_run 2>/dev/null || true")
time.sleep(1)

# Re-run dry runs
print()
print("=== dry_run_dataset ===")
rc1 = stream("cd %s && python3 Dry_Run/dry_run_dataset.py 2>&1" % CODE, timeout=180)
print("\nexit=%d" % rc1)

print()
print("=== dry_run_cpu_only ===")
rc2 = stream("cd %s && python3 Dry_Run/dry_run_cpu_only.py 2>&1" % CODE, timeout=180)
print("\nexit=%d" % rc2)

print()
print("=== dry_run_gpu_cpu ===")
rc3 = stream("cd %s && HF_HOME=/workspace/.hf_cache python3 Dry_Run/dry_run_gpu_cpu.py 2>&1" % CODE, timeout=300)
print("\nexit=%d" % rc3)

# Launch pipeline only if dry runs pass
if rc1 == 0 and rc2 == 0 and rc3 == 0:
    print()
    print("=== ALL DRY RUNS PASSED — launching GPU pipeline ===")
    PIPELINE = (
        "cd {code} && "
        "export HF_HOME=/workspace/.hf_cache && "
        "export TRANSFORMERS_CACHE=/workspace/.hf_cache/hub && "
        "mkdir -p /workspace/logs && "
        "python3 GPU_CPU/osm_behavioral.py && "
        "python3 GPU_CPU/cdva_patching.py && "
        "python3 GPU_CPU/cdva_calibration.py && "
        "python3 CPU_Only/leaderboard.py && "
        "echo GPU_PIPELINE_DONE"
    ).format(code=CODE)
    run("tmux new-session -d -s gpu_run -x 250 -y 50")
    run("tmux send-keys -t gpu_run 'bash -c \"%s\" 2>&1 | tee /workspace/logs/gpu_run.log' Enter" % PIPELINE)
    time.sleep(2)
    _, sess = run("tmux ls 2>&1")
    print("tmux:", sess)
    print()
    print("GPU PIPELINE LAUNCHED in tmux 'gpu_run'")
    print("ssh root@%s -p %s  then  tmux attach -t gpu_run" % (HOST, PORT))
else:
    print()
    # Show what the actual errors are
    print("DRY RUNS FAILED: dataset=%d cpu=%d gpu=%d" % (rc1, rc2, rc3))
    print("Collecting error details ...")
    for label, cmd in [
        ("dataset", "cd %s && python3 Dry_Run/dry_run_dataset.py 2>&1 | head -30" % CODE),
        ("cpu", "cd %s && python3 Dry_Run/dry_run_cpu_only.py 2>&1 | head -30" % CODE),
    ]:
        _, err = run(cmd, timeout=60)
        print("\n--- %s ---\n%s" % (label, err))

c.close()
