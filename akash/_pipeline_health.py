"""Comprehensive pipeline health + progress audit."""
import json
import re
import sys
from datetime import datetime, timezone

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"

EXPECTED_SEEDS = 596
EXPECTED_ROWS = EXPECTED_SEEDS * 12  # 7152


def run(c, cmd, timeout=60):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    return (o.read() + e.read()).decode("utf-8", "replace").strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)

    print("=" * 60)
    print("PIPELINE HEALTH AUDIT", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    print("=" * 60)

    issues = []

    # ── Markers ──
    print("\n[1] STATE MARKERS")
    markers = ["INSTALL_OK", "PREDOWNLOAD_OK", "DATASET_OK", "GPU_PIPELINE_OK", "PIPELINE_COMPLETE"]
    marker_state = {}
    for m in markers:
        exists = run(c, f"test -f /data/state/{m} && echo YES || echo NO") == "YES"
        marker_state[m] = exists
        print(f"  {m}: {'OK' if exists else 'MISSING'}")

    # ── Processes ──
    print("\n[2] RUNNING PROCESSES")
    procs = run(c, "pgrep -af 'supervise_pipeline|_full_pipeline|run_gpu_pipeline|regenerate_api' || echo NONE")
    print(procs[:1500] if procs else "  NONE")
    gpu_running = "run_gpu_pipeline.py" in procs
    regen_running = "regenerate_api_slots" in procs

    # ── Pentad integrity ──
    print("\n[3] PENTAD DATASET")
    pentad_check = run(
        c,
        "cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python - <<'PY'\n"
        "import json, pandas as pd\n"
        "from Dataset.validate_pentad import assert_production_ready, _PENTAD_PATH\n"
        "df = pd.read_parquet(_PENTAD_PATH)\n"
        "audit = df[df['seed_source'].astype(str).str.lower().isin({'bbq','crows_pairs','stereoset'})]\n"
        "print('rows', len(audit))\n"
        "print('seeds', audit['seed_id'].nunique())\n"
        "print('slots', audit['slot'].value_counts().to_dict())\n"
        "try:\n"
        "    assert_production_ready(df)\n"
        "    print('VALID', 'yes')\n"
        "except Exception as e:\n"
        "    print('VALID', 'no')\n"
        "    print('ERR', str(e))\n"
        "mp = __import__('pathlib').Path('Dataset/seeds/pentad_manifest.json')\n"
        "if mp.exists():\n"
        "    print('MANIFEST', mp.read_text()[:200])\n"
        "PY",
        timeout=90,
    )
    print(pentad_check)

    pentad_valid = "VALID yes" in pentad_check.replace("\n", " ") or "VALID\nyes" in pentad_check
    row_match = re.search(r"rows (\d+)", pentad_check)
    seed_match = re.search(r"seeds (\d+)", pentad_check)
    n_rows = int(row_match.group(1)) if row_match else 0
    n_seeds = int(seed_match.group(1)) if seed_match else 0

    if n_rows != EXPECTED_ROWS or n_seeds != EXPECTED_SEEDS:
        issues.append(f"Pentad count wrong: {n_rows} rows, {n_seeds} seeds (expected {EXPECTED_ROWS}/{EXPECTED_SEEDS})")
    if not pentad_valid:
        issues.append("Pentad failed assert_production_ready()")

    # ── GPU results ──
    print("\n[4] GPU RESULTS")
    results_info = run(
        c,
        "ls -la /data/Audit_Benchmark/Code/mirage/results/*.parquet 2>/dev/null || echo NO_PARQUET\n"
        "cd /data/Audit_Benchmark/Code/mirage && /data/venv/bin/python - <<'PY' 2>/dev/null || true\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "for p in ['results/behavioral_results.parquet','results/cdva_results.parquet']:\n"
        "    f = Path(p)\n"
        "    if f.exists():\n"
        "        df = pd.read_parquet(f)\n"
        "        print(p, 'rows', len(df), 'models', df['model_name'].nunique() if 'model_name' in df.columns else '?')\n"
        "        if 'success_flag' in df.columns:\n"
        "            print('  success_rate', df['success_flag'].mean())\n"
        "    else:\n"
        "        print(p, 'MISSING')\n"
        "PY",
    )
    print(results_info)

    # ── Pipeline log progress ──
    print("\n[5] PIPELINE LOG (latest progress)")
    log_tail = run(c, "tail -25 /data/logs/pipeline_attempt_1.log 2>/dev/null || tail -25 /data/logs/pipeline_attempt_*.log 2>/dev/null | tail -25")
    print(log_tail[-2500:])

    # Parse progress from log
    progress_lines = [l for l in log_tail.split("\n") if "prompts done" in l]
    latest_progress = progress_lines[-1] if progress_lines else ""

    # Extract model and count
    prog_match = re.search(r"OSM ([^:]+): (\d+)/(\d+) prompts done \(sample_index=(\d+)", latest_progress)
    current_model = prog_match.group(1) if prog_match else "unknown"
    done_n = int(prog_match.group(2)) if prog_match else 0
    total_n = int(prog_match.group(3)) if prog_match else 7152
    sample_idx = int(prog_match.group(4)) if prog_match else 0

    # Check which step from log
    step_info = run(c, "grep -E 'Step [0-9]/4|GPU PIPELINE COMPLETE|CDVA|tau calibration|FAILED|Error|Traceback' /data/logs/pipeline_attempt_1.log 2>/dev/null | tail -15")
    print("\n[6] PIPELINE STEPS (recent milestones)")
    print(step_info or "  (none)")

    # Only flag errors after the latest GPU Step 2 start (ignore historical failures)
    gpu_step2_line = run(c, 'grep -n "Step 2/4: Behavioral" /data/logs/pipeline_attempt_1.log | tail -1')
    recent_errors = ""
    if gpu_step2_line:
        ln = gpu_step2_line.split(":")[0]
        recent_errors = run(
            c,
            f"tail -n +{ln} /data/logs/pipeline_attempt_1.log | "
            "grep -iE 'Traceback|FAILED|ERROR:' | grep -v 'failure_reason' | head -5",
        )
    if recent_errors.strip():
        issues.append(f"Errors since current GPU run: {recent_errors[:200]}")

    # ── GPU ──
    print("\n[7] GPU")
    print(run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null"))

    # ── Stale checkpoints ──
    print("\n[8] STALE CHECKPOINTS")
    ckpt = run(c, "ls -la /data/Audit_Benchmark/Code/mirage/Dataset/seeds/*checkpoint* 2>/dev/null || echo none")
    print(ckpt)

    c.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if issues:
        print("ISSUES FOUND:")
        for i in issues:
            print(f"  - {i}")
    else:
        print("No data integrity issues detected.")

    print(f"\nMarker state: DATASET_OK={marker_state.get('DATASET_OK')}, GPU_OK={marker_state.get('GPU_PIPELINE_OK')}, COMPLETE={marker_state.get('PIPELINE_COMPLETE')}")
    print(f"GPU pipeline running: {gpu_running}")
    print(f"Latest: model={current_model}, progress={done_n}/{total_n}, sample_index={sample_idx}")

    # ETA calculation
    if gpu_running and done_n > 0 and not marker_state.get("PIPELINE_COMPLETE"):
        # 4 models, each: 7152 det + 596*5 variance (slot-a only) = 7152 + 2980 = 10132 per model? 
        # Actually variance is only slot-a: 596 prompts × 5 samples = 2980 per model
        # Deterministic: 7152 per model × 4 models = 28608
        # Variance: 596 × 5 × 4 = 11920
        # Total prompt evaluations: 28608 + 11920 = 40528 ... but they're sequential

        # From log rate: estimate from recent progress
        rate_per_min = None
        if len(progress_lines) >= 2:
            # rough: use last line only with ~2 prompts/sec from earlier observation (~120/min)
            pass

        # Per model deterministic: 7152 prompts
        # At ~150 prompts/min observed earlier: 7152/150 = 47.7 min per model det
        # Variance 596*5 = 2980 at similar rate: 2980/150 = 20 min per model
        # Per model total: ~68 min × 4 = 272 min = 4.5 hours
        # CDVA: additional time
        # Current: model 1 det pass at done_n/7152

        models = ["llama-3.1-8b-instruct", "phi-4-mini-instruct", "qwen3-4b-instruct", "gemma-3-4b-it"]
        try:
            model_idx = models.index(current_model.strip())
        except ValueError:
            model_idx = 0

        prompts_per_min = 150  # conservative from log (~800 in 5 min)

        if sample_idx == 0:
            remaining_det_this_model = total_n - done_n
            remaining_det_other_models = (3 - model_idx) * 7152
            remaining_var_all = 4 * 596 * 5  # 11920
            remaining_prompts = remaining_det_this_model + remaining_det_other_models + remaining_var_all
        else:
            # in variance pass
            var_total = 596 * 5
            remaining_var_this = var_total - done_n if sample_idx <= 5 else 0
            remaining_var_other_samples = max(0, (5 - sample_idx) * 596)
            remaining_var_other_models = (3 - model_idx) * 596 * 5
            remaining_prompts = remaining_var_this + remaining_var_other_samples + remaining_var_other_models + 0

        hours_left = remaining_prompts / prompts_per_min / 60
        cdva_hours = 1.5  # CDVA estimate
        total_hours = hours_left + cdva_hours

        print(f"\nETA (approximate): {total_hours:.1f} hours remaining")
        print(f"  ({remaining_prompts} prompts @ ~{prompts_per_min}/min + ~{cdva_hours}h CDVA/calibration)")

    return issues, marker_state, gpu_running, prog_match is not None


if __name__ == "__main__":
    main()
