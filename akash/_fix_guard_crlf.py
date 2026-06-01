"""Fix CRLF on shell scripts on VM."""
import paramiko

HOST, PORT, USER, PW = "provider.a100.dsm.val.akash.pub", 31532, "root", "MirageVM2026!"
CMD = """
sed -i 's/\\r$//' /data/Audit_Benchmark/akash/autonomous_guard.sh
sed -i 's/\\r$//' /data/Audit_Benchmark/akash/supervise_pipeline.sh
bash /data/Audit_Benchmark/akash/autonomous_guard.sh
echo guard_exit=$?
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, PORT, username=USER, password=PW, timeout=30)
_, o, _ = c.exec_command(CMD, timeout=30)
print(o.read().decode())
c.close()
