#!/bin/bash

clear

echo "==== Incializando ===="

sleep 1

echo "==== Configurando o Ollama ===="

curl -fsSL https://ollama.com/install.sh | sh

sleep 5

nohup ollama serve &

sleep 2

echo "Llama3.2:"

ollama pull llama3.2

echo "Deepseek-r1"
ollama pull deepseek-r1:latest

echo "mxbai-embed-large"
ollama pull mxbai-embed-large:latest

sleep 2

clear

ollama ls


source env/bin/activate

clear

echo "==== Iniciando aplicação ===="
nohup waitress-serve --listen=0.0.0.0:8000 main_router:app &