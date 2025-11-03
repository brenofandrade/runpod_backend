#!/usr/bin/env bash
set -euo pipefail

# carrega variáveis do .env (incluindo POD_ID e TEAMS_WEBHOOK_URL se quiser)
source ../.env

LOG_FILE="../log/watchdog.log"

POD_ID="${POD_ID:?POD_ID not set}"

# URL que vamos CHECAR
APP_HEALTH_URL="https://${POD_ID}-8000.proxy.runpod.net/health"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}


while true; do
  http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$APP_HEALTH_URL")

  if [[ "$http_code" == "200" ]]; then
    log "OK: healthcheck succeeded"
  else
    log "Healthcheck FAILED. HTTP status: $http_code"
  fi

  sleep 5
done
