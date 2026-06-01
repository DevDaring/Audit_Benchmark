"""Poll VM until regen completes, then validate and restart supervisor."""
import sys
import time

import paramiko

HOST, PORT, PW = "provider.a100.dsm.val.akash.pub", 31532, "MirageVM2026!"

POLL = r"""
pgrep -f 'regenerate_api_slots.py' >/dev/null && echo RUNNING || echo DONE
test -f /data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json && \
  python3 -c "import json; print('ctx_ckpt', len(json.load(open('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/context_shift_checkpoint.json'))))" \
  || echo ctx_ckpt 0
/data/venv/bin/python - <<'PY'
import pandas as pd
df = pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet')
print('rows', len(df), 'de', int(df.slot.isin(['d','e']).sum()))
PY
pgrep -af supervise_pipeline || true
"""

FINISH = open(__import__("pathlib").Path(__file__).parent / "_finish_slotb_repair.py").read()
# extract FINISH script from _finish_slotb_repair.py
FINISH_SCRIPT = FINISH.split('FINISH = r"""')[1].split('"""')[0]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username="root", password=PW, timeout=30, banner_timeout=60)
    return c


def run(c, script, timeout=120):
    _, stdout, stderr = c.exec_command(f"bash -s << 'EOF'\n{script}\nEOF", timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    deadline = time.time() + 7200
    last = ""
    while time.time() < deadline:
        try:
            c = connect()
            # keep supervisor stopped during regen
            run(c, "pkill -f supervise_pipeline.sh 2>/dev/null || true; pkill -f _full_pipeline.py 2>/dev/null || true", 30)
            _, out, _ = run(c, POLL, 60)
            c.close()
        except Exception as exc:
            print(f"[poll] reconnect error: {exc}")
            time.sleep(20)
            continue

        if out != last:
            print(time.strftime("%H:%M:%S"), out.strip())
            last = out

        if "DONE" in out and "rows 7152" in out and "de 4768" in out:
            print("Regen complete (7152 rows).")
            break
        if "DONE" in out and "de 0" not in out.split("de")[1].split()[0:1]:
            # partial - keep waiting if process done but rows wrong
            parts = out.split()
            try:
                de = int([p for p in parts if p.startswith("de") or p == "de"][0].replace("de", "") or parts[parts.index("de")+1])
            except Exception:
                de = 0
            if "DONE" in out and de >= 4768:
                break
        time.sleep(45)

    print("\n=== Finalize ===")
    c = connect()
    run(c, "pkill -f supervise_pipeline.sh 2>/dev/null || true; pkill -f _full_pipeline.py 2>/dev/null || true", 30)
    code, out, err = run(c, FINISH_SCRIPT, 600)
    print(out)
    if err.strip():
        print("STDERR:", err[-4000:])
    c.close()
    if code != 0:
        sys.exit(code)
    print("Pipeline restarted with validated pentad.")


if __name__ == "__main__":
    main()
