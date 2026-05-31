"""Fix missing packages and re-run the full pipeline."""
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

# Install missing packages
print("=== Installing missing packages ===")
rc, out = run("pip install python-dotenv outlines boto3 google-generativeai mistralai openai scipy paramiko 2>&1 | tail -5")
print(out)
print("exit=%d" % rc)

# Verify
rc, out = run("python3 -c 'import dotenv, outlines, boto3; print(\"OK\")'")
print("Verify:", out)

# Kill bad gpu_run session (it was launched before fix)
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

# Launch GPU pipeline only if all dry runs pass
if rc1 == 0 and rc2 == 0 and rc3 == 0:
    print()
    print("=== ALL DRY RUNS PASSED — launching GPU pipeline ===")
    PIPELINE = (
        "cd %s && "
        "export HF_HOME=/workspace/.hf_cache && "
        "python3 GPU_CPU/osm_behavioral.py && "
        "python3 GPU_CPU/cdva_patching.py && "
        "python3 GPU_CPU/cdva_calibration.py && "
        "python3 CPU_Only/leaderboard.py && "
        "echo GPU_PIPELINE_DONE"
    ) % CODE
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
    print("DRY RUNS FAILED: dataset=%d cpu=%d gpu=%d — pipeline NOT launched" % (rc1, rc2, rc3))

c.close()
