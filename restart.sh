#!/bin/bash

echo "==== Configurando o Ollama ===="

curl -fsSL https://ollama.com/install.sh | sh

sleep 5

nohup ollama serve &

sleep 2

ollama pull llama3.2
ollama pull deepseek-r1:latest
ollama pull mxbai-embed-large:latest

sleep 2

source env/bin/activate

clear

echo "==== Iniciando aplicação ===="
nohup waitress-serve --listen=0.0.0.0:8000 main:app &