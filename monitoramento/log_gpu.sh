#!/usr/bin/env bash
LOG_FILE="../../logs/gpu_usage.log"

# cabeçalho opcional
echo "timestamp,util_gpu(%),util_mem(%),mem_used(MiB),mem_total(MiB)" >> "$LOG_FILE"

while true; do
  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader,nounits >> "$LOG_FILE"

  sleep 5   # intervalo em segundos
done

# chmod +x log_gpu.sh
# ./log_gpu.sh
# nohup ./log_gpu.sh >/dev/null 2>&1 &