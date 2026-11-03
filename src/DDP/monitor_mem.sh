#!/bin/bash
# Cada 2 s, en el nodo donde corre: memoria del job según su cgroup (lo que dispara el OOM de Slurm),
# RAM total usada del nodo, RSS sumado de los python del usuario y uso de /dev/shm.
cg=/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup)
while [ "$cg" != "/sys/fs/cgroup" ] && [[ "$(basename "$cg")" != job_* ]]; do
    cg=$(dirname "$cg")
done
while true; do
    job=$([ -f "$cg/memory.current" ] && awk '{printf "%.1f", $1 / 2^30}' "$cg/memory.current" || echo "?")
    nodo=$(free -g | awk '/Mem:/ {print $3}')
    rss=$(ps -u "$USER" -o rss=,comm= | awk '$2 == "python" {s += $1} END {printf "%.1f", s / 2^20}')
    shm=$(df -BG --output=used /dev/shm | tail -1 | tr -d ' ')
    echo "[MEM] $(hostname -s) $(date +%T) job=${job}G nodo=${nodo}G python_rss=${rss}G shm=${shm}"
    sleep 2
done
