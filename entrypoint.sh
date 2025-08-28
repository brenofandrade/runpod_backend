#!/usr/bin/env bash
set -euo pipefail

# Inicia o daemon do Ollama em background
echo "[entrypoint] iniciando ollama serve em background..."
(ollama serve > /tmp/ollama.log 2>&1) &

# Espera o Ollama ficar pronto
echo "[entrypoint] aguardando ollama ficar pronto em ${OLLAMA_HOST:-0.0.0.0:11434}..."
for i in {1..60}; do
  if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null; then
    echo "[entrypoint] ollama OK"
    break
  fi
  sleep 1
  if [ "$i" -eq 60 ]; then
    echo "[entrypoint] ERRO: ollama não ficou pronto a tempo."
    echo "---- /tmp/ollama.log ----"; tail -n +1 /tmp/ollama.log || true
    exit 1
  fi
done

# Normaliza lista de modelos: vírgula ou espaço
MODELS_RAW="${OLLAMA_MODELS:-}"
MODELS="$(echo "$MODELS_RAW" | tr ',' ' ' | xargs)"

if [ -n "$MODELS" ]; then
  echo "[entrypoint] preparando modelos: $MODELS"
  # Faz pull de cada modelo se ainda não estiver disponível
  for M in $MODELS; do
    if ! ollama list | awk '{print $1}' | grep -qx "$M"; then
      echo "[entrypoint] puxando modelo: $M"
      if ! ollama pull "$M"; then
        echo "[entrypoint] AVISO: falha ao puxar '$M' (segue sem abortar)."
      fi
    else
      echo "[entrypoint] modelo já presente: $M"
    fi
  done
else
  echo "[entrypoint] OLLAMA_MODELS vazio — seguindo sem pull."
fi

# Define modelo padrão para o backend se não foi definido
if [ -z "${GENERATIVE_MODEL:-}" ]; then
  if [ -n "$MODELS" ]; then
    export GENERATIVE_MODEL="$(echo "$MODELS" | awk '{print $1}')"
    echo "[entrypoint] GENERATIVE_MODEL não definido — usando primeiro da lista: $GENERATIVE_MODEL"
  else
    echo "[entrypoint] AVISO: nenhum GENERATIVE_MODEL definido e OLLAMA_MODELS vazio."
  fi
fi

# Mostra info útil
echo "[entrypoint] OLLAMA_HOST=${OLLAMA_HOST:-}"
echo "[entrypoint] GENERATIVE_MODEL=${GENERATIVE_MODEL:-<vazio>}"
echo "[entrypoint] iniciando backend Flask na porta ${PORT:-8000}..."

# Sobe o Flask via waitress (main:app)
exec waitress-serve --listen=0.0.0.0:${PORT:-8000} main:app
