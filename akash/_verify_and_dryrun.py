"""
Verify all packages on the VM, then launch a 2-seed GPU dry run.
"""
import paramiko, sys, time

VM_HOST = "provider.a100.dsm.val.akash.pub"
VM_PORT = 30594
DRY_LOG = "/workspace/dry_run_2seeds.log"
POLL = 15
MAX_MIN = 40


def conn():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(VM_HOST, port=VM_PORT, username="root", password="MirageVM2026!",
              timeout=30, banner_timeout=60)
    return c


def run(c, cmd, t=20):
    _, o, e = c.exec_command(cmd, timeout=t)
    return (o.read().decode(errors="replace") + e.read().decode(errors="replace")).strip()


def sp(s):
    try:
        print(s)
    except UnicodeEncodeError:
        print(s.encode("ascii", errors="replace").decode())


# ---- 1. Verify packages ----
print("=" * 50)
print("PACKAGE VERIFICATION")
print("=" * 50)
c = conn()
checks = [
    ("torch",          "import torch; print(torch.__version__, torch.cuda.is_available())"),
    ("dotenv",         "import dotenv; print(dotenv.__version__)"),
    ("flash_attn",     "import flash_attn; print(flash_attn.__version__)"),
    ("transformers",   "import transformers; print(transformers.__version__)"),
    ("transformer_lens", "import transformer_lens; print('OK')"),
    ("nnsight",        "import nnsight; print('OK')"),
    ("pandas",         "import pandas; print(pandas.__version__)"),
    ("pyarrow",        "import pyarrow; print(pyarrow.__version__)"),
    ("outlines",       "import outlines; print('OK')"),
]
all_ok = True
for name, code in checks:
    r = run(c, f"python3 -c '{code}' 2>&1", t=30)
    ok = "Error" not in r and "Traceback" not in r
    sp(f"  {'OK  ' if ok else 'FAIL'} {name:<22} {r[:60]}")
    if not ok:
        all_ok = False
c.close()

if not all_ok:
    print("\nSome packages FAILED. Cannot proceed with dry run.")
    sys.exit(1)

print("\nAll packages verified.\n")

# ---- 2. Launch 2-seed dry run ----
print("=" * 50)
print("LAUNCHING 2-SEED DRY RUN")
print("=" * 50)

c = conn()
run(c, "tmux kill-session -t dry2 2>/dev/null; true")

run_cmd = (
    "cd /workspace/Audit_Benchmark/Code/mirage && "
    "PYTHONPATH=/workspace/Audit_Benchmark/Code/mirage "
    "python3 Dry_Run/dry_run_gpu_cpu.py --n-seeds 2"
    " 2>&1 | tee " + DRY_LOG +
    "; echo DRY_RUN_DONE >> " + DRY_LOG
)
tmux_cmd = "tmux new-session -d -s dry2 '" + run_cmd + "'"
sp(run(c, tmux_cmd, t=15))
time.sleep(2)
sp(run(c, "tmux list-sessions 2>&1", t=5))
c.close()
print(f"Dry run started. Log: {DRY_LOG}\n")

# ---- 3. Poll ----
deadline = time.time() + MAX_MIN * 60
polls = 0
while time.time() < deadline:
    time.sleep(POLL)
    polls += 1
    try:
        c2 = conn()
        done = run(c2, f"grep -c DRY_RUN_DONE {DRY_LOG} 2>/dev/null || echo 0")
        tail = run(c2, f"tail -6 {DRY_LOG}")
        c2.close()
    except Exception as ex:
        sp(f"  [{polls*POLL}s] reconnect: {ex}")
        continue
    sp(f"[{polls*POLL}s] DONE={done.split()[-1]}")
    for ln in tail.splitlines():
        sp("  " + ln)
    if done.strip().endswith("1"):
        break
else:
    print("TIMEOUT.")
    sys.exit(1)

# ---- 4. Print results ----
c = conn()
print("\n" + "=" * 50)
print("DRY RUN LOG (full):")
print("=" * 50)
sp(run(c, f"cat {DRY_LOG}", t=60))

fails = int(run(c, f"grep -c ' FAIL ' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
passes = int(run(c, f"grep -c ' PASS ' {DRY_LOG} 2>/dev/null || echo 0").split()[-1])
c.close()

print(f"\nRESULT: {passes} PASS, {fails} FAIL")
if fails == 0:
    print("2-seed dry run PASSED.")
else:
    print("Dry run had FAILs. Review log above.")
    sys.exit(1)
