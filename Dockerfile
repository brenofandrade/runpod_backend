# ========= Base =========
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependências de sistema (curl para instalar ollama)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg \
 && rm -rf /var/lib/apt/lists/*

# ========= Ollama =========
# Instala o Ollama (CPU/GPU será detectado no runtime com --gpus all)
RUN curl -fsSL https://ollama.com/install.sh | sh

# ========= App =========
WORKDIR /app

# Copia requirements e instala
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copia o backend
COPY main.py ./

# Copia o entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# ========= Config padrão =========
# Porta do Flask e do Ollama
ENV PORT=8000 \
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=24h \
    # Liste os modelos a carregar (separados por vírgula)
    OLLAMA_MODELS="llama3.2,deepseek-r1,gpt-oss,llama4" \
    # Opcional: nome do modelo preferido para o backend (se vazio, usa o 1º da lista)
    GENERATIVE_MODEL=""

EXPOSE 8000 11434

# Healthcheck do Ollama (fica saudável quando /api/tags responde)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
 CMD curl -sf http://127.0.0.1:11434/api/tags >/dev/null || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]
