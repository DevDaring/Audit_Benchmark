#!/usr/bin/env bash
# P0: resource watchdog — logs every 10s to PERSISTENT storage (/data/logs).
# Survives container eviction. After a restart: tail -50 /data/logs/watchdog.log
# to see exactly what was spiking at the moment of eviction.
mkdir -p /data/logs
LOG=/data/logs/watchdog.log

echo "=== watchdog boot $(date -u +%FT%TZ) pid1_age=$(ps -p 1 -o etimes= 2>/dev/null | tr -d ' ')s ===" >> "$LOG"

while true; do
  ts=$(date -u +%FT%TZ)
  mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo '?')
  memmax=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo '?')
  load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo '?')

  # EPHEMERAL disk — the suspected eviction trigger
  eph=$(df -h /workspace 2>/dev/null | awk 'NR==2{print $3"/"$2" ("$5")"}')
  eph_root=$(df -h / 2>/dev/null | awk 'NR==2{print $3"/"$2" ("$5")"}')
  eph_inodes=$(df -i / 2>/dev/null | awk 'NR==2{print "inodes:"$5}')

  # PERSISTENT disk (/data volume)
  dat=$(df -h /data 2>/dev/null | awk 'NR==2{print $3"/"$2" ("$5")"}' || echo 'not mounted')

  # GPU
  gpu=$(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | head -1 || echo 'n/a')

  age=$(ps -p 1 -o etimes= 2>/dev/null | tr -d ' ')

  echo "$ts pid1=${age}s mem=${mem}/${memmax} load=[${load}] root=${eph_root} ${eph_inodes} workspace=${eph} data=${dat} gpu=${gpu}" >> "$LOG"
  sleep 10
done
