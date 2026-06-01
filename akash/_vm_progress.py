"""Full VM progress audit."""
import json
import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"


def run(client, cmd: str) -> str:
    _, o, e = client.exec_command(cmd, timeout=60)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    return out + (f"\n[stderr] {err}" if err.strip() else "")


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=25)

    checks = [
        ("state markers", "ls -la /data/state/ 2>/dev/null; for f in INSTALL_OK PREDOWNLOAD_OK DATASET_OK GPU_PIPELINE_OK PIPELINE_COMPLETE; do [ -f /data/state/$f ] && echo OK:$f || echo MISSING:$f; done"),
        ("pentad rows", "/data/venv/bin/python -c \"import pandas as pd; df=pd.read_parquet('/data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_dataset.parquet'); print('rows',len(df)); print('seeds',df.seed_id.nunique()); print(df.slot.value_counts().to_dict())\""),
        ("manifest", "cat /data/Audit_Benchmark/Code/mirage/Dataset/seeds/pentad_manifest.json 2>/dev/null || echo no manifest"),
        ("excluded", "cat /data/Audit_Benchmark/Code/mirage/Dataset/seeds/excluded_seeds.json 2>/dev/null | head -20"),
        ("assert ready", "cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python -c \"import pandas as pd; from Dataset.validate_pentad import assert_production_ready; assert_production_ready(pd.read_parquet('Dataset/seeds/pentad_dataset.parquet')); print('PRODUCTION READY')\""),
        ("gpu results", "ls -la /data/Audit_Benchmark/Code/mirage/results/ 2>/dev/null; ls -la /data/Audit_Benchmark/Code/mirage/results/*.parquet 2>/dev/null || echo no parquet yet"),
        ("pipeline log", "tail -3 /data/logs/pipeline_attempt_1.log 2>/dev/null"),
        ("supervisor", "pgrep -af supervise_pipeline || pgrep -af _full_pipeline || echo no pipeline running"),
        ("gpu", "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo no gpu"),
        ("recent logs", "tail -8 /data/logs/pipeline_attempt_*.log 2>/dev/null || tail -8 /data/Audit_Benchmark/LOG/regen_api_slots.log"),
    ]

    for title, cmd in checks:
        print(f"\n=== {title} ===")
        print(run(c, cmd).strip())

    c.close()


if __name__ == "__main__":
    main()
