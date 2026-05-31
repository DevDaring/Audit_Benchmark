"""
akash/check_vm.py — Verify that the Akash VM is healthy and MIRAGE-ready.

Usage:
    python akash/check_vm.py --host <IP> --port <SSH_PORT>

Checks performed:
    1. SSH connectivity
    2. CUDA / GPU presence (nvidia-smi)
    3. Python 3.10 available
    4. All required packages installed (torch, flash_attn, transformer_lens, etc.)
    5. .env present with required keys
    6. Repo cloned and folder structure intact
    7. Dry-run passes (Dataset + CPU_Only; GPU dry-run optional via --full)

The script prints a summary table and exits with code 0 on full pass.
"""

import argparse
import sys
import time
from typing import Optional

try:
    import paramiko  # type: ignore
except ImportError:
    print("[check_vm] paramiko not installed. Run: pip install paramiko")
    sys.exit(1)


# ------------------------------------------------------------------ helpers

def _ssh_connect(host: str, port: int, user: str = "root",
                 password: str = "MirageVM2026!",
                 timeout: int = 30) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username=user, password=password,
                   timeout=timeout, banner_timeout=60)
    return client


def _run(client: paramiko.SSHClient, cmd: str,
         timeout: int = 120) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(), stderr.read().decode()


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✓" if ok else "✗"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return ok


# ------------------------------------------------------------------ checks

CHECK_PACKAGES = [
    ("torch",            "import torch; print(torch.__version__, torch.cuda.is_available())"),
    ("flash_attn",       "import flash_attn; print(flash_attn.__version__)"),
    ("transformer_lens", "import transformer_lens; print(transformer_lens.__version__)"),
    ("nnsight",          "import nnsight; print(getattr(nnsight,'__version__','ok'))"),
    ("outlines",         "import outlines; print(getattr(outlines,'__version__','ok'))"),
    ("transformers",     "import transformers; print(transformers.__version__)"),
    ("openai",           "import openai; print(openai.__version__)"),
    ("boto3",            "import boto3; print(boto3.__version__)"),
    ("google-generativeai", "import google.generativeai as g; print(getattr(g,'__version__','ok'))"),
    ("mistralai",        "import mistralai; print(getattr(mistralai,'__version__','ok'))"),
    ("pandas",           "import pandas; print(pandas.__version__)"),
]

REQUIRED_ENV_KEYS = [
    "DEEPSEEK_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "HF_TOKEN",
]

REQUIRED_DIRS = [
    "/workspace/Audit_Benchmark/Code/mirage",
    "/workspace/Audit_Benchmark/Code/mirage/GPU_CPU",
    "/workspace/Audit_Benchmark/Code/mirage/CPU_Only",
    "/workspace/Audit_Benchmark/Code/mirage/Dataset",
    "/workspace/Audit_Benchmark/Code/mirage/Dry_Run",
    "/workspace/Audit_Benchmark/Code/mirage/Results",
]


def run_checks(host: str, port: int, run_gpu_dry: bool = False) -> bool:
    print(f"\n[check_vm] Connecting to {host}:{port} ...")

    # 1. connectivity
    try:
        client = _ssh_connect(host, port)
        _check("SSH connectivity", True)
    except Exception as exc:
        _check("SSH connectivity", False, str(exc))
        return False

    all_ok = True

    # 2. GPU
    rc, out, _ = _run(client, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    gpu_ok = rc == 0 and "A100" in out
    all_ok &= _check("NVIDIA A100 GPU", gpu_ok, out.strip()[:80])

    # 3. Python
    rc, out, _ = _run(client, "python3 --version")
    py_ok = rc == 0 and "3.10" in out
    all_ok &= _check("Python 3.10", py_ok, out.strip())

    # 4. packages
    print("\n  [packages]")
    for pkg, cmd in CHECK_PACKAGES:
        rc, out, err = _run(client, f"python3 -c \"{cmd}\"")
        ok = rc == 0
        all_ok &= _check(f"    {pkg:<25}", ok, out.strip()[:50] if ok else err.strip()[:50])

    # 5. .env keys
    print("\n  [.env keys]")
    for key in REQUIRED_ENV_KEYS:
        rc, out, _ = _run(
            client,
            f"grep -q '^{key}=' /workspace/Audit_Benchmark/Code/mirage/.env && echo yes || echo no"
        )
        present = out.strip() == "yes"
        all_ok &= _check(f"    {key:<30}", present)

    # 6. repo structure
    print("\n  [directories]")
    for d in REQUIRED_DIRS:
        rc, _, _ = _run(client, f"test -d {d}")
        all_ok &= _check(f"    {d}", rc == 0)

    # 7. Dataset dry run
    print("\n  [dry runs]")
    rc, out, err = _run(
        client,
        "cd /workspace/Audit_Benchmark/Code/mirage && "
        "python3 Dry_Run/dry_run_dataset.py 2>&1 | tail -5",
        timeout=180,
    )
    all_ok &= _check("  dry_run_dataset.py", rc == 0, out.strip().splitlines()[-1] if out else err[:80])

    rc, out, err = _run(
        client,
        "cd /workspace/Audit_Benchmark/Code/mirage && "
        "python3 Dry_Run/dry_run_cpu_only.py 2>&1 | tail -5",
        timeout=180,
    )
    all_ok &= _check("  dry_run_cpu_only.py", rc == 0, out.strip().splitlines()[-1] if out else err[:80])

    if run_gpu_dry:
        rc, out, err = _run(
            client,
            "cd /workspace/Audit_Benchmark/Code/mirage && "
            "python3 Dry_Run/dry_run_gpu_cpu.py 2>&1 | tail -5",
            timeout=300,
        )
        all_ok &= _check("  dry_run_gpu_cpu.py", rc == 0, out.strip().splitlines()[-1] if out else err[:80])

    client.close()

    print("\n" + ("=" * 60))
    status = "PASS" if all_ok else "FAIL"
    print(f"  Overall: {status}")
    print("=" * 60)
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MIRAGE Akash VM health")
    parser.add_argument("--host", required=True, help="VM IP address")
    parser.add_argument("--port", type=int, default=22, help="SSH port (default 22)")
    parser.add_argument("--full", action="store_true",
                        help="Also run the GPU dry run (slow)")
    args = parser.parse_args()

    ok = run_checks(args.host, args.port, run_gpu_dry=args.full)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
