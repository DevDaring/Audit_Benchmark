"""Upload corrected MIRAGE dataset/scoring files to the Akash VM."""
import pathlib
import sys

import paramiko

HOST = "provider.a100.dsm.val.akash.pub"
PORT = 31532
USER = "root"
PASSWORD = "MirageVM2026!"

ROOT = pathlib.Path(__file__).resolve().parents[1]
REMOTE = "/data/Audit_Benchmark/Code/mirage"

FILES = [
    "Dataset/gold_utils.py",
    "Dataset/prompt_utils.py",
    "Dataset/pentad_generator.py",
    "Dataset/sample_seeds.py",
    "Dataset/validate_pentad.py",
    "Dataset/context_shift_drafter.py",
    "Dataset/cot_attack_generator.py",
    "patch_det_slots.py",
    "regenerate_api_slots.py",
    "run_dataset.py",
    "CPU_Only/scoring.py",
    "GPU_CPU/run_gpu_pipeline.py",
    "GPU_CPU/pipeline_guards.py",
]

REMOTE_FILES = [
    ("akash/_full_pipeline.py", "/data/Audit_Benchmark/akash/_full_pipeline.py"),
]


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()
    for rel in FILES:
        local = ROOT / "Code" / "mirage" / rel.replace("/", "\\").replace("\\", "/")
        remote = f"{REMOTE}/{rel}"
        print(f"upload {rel}")
        sftp.put(str(local), remote)
    for local_rel, remote in REMOTE_FILES:
        local = ROOT / local_rel.replace("/", "\\").replace("\\", "/")
        print(f"upload {local_rel}")
        sftp.put(str(local), remote)
    sftp.close()
    client.close()
    print("done")


if __name__ == "__main__":
    main()
