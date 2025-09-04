#!/bin/bash

echo "==== Incializando ===="


echo "==== Criando ambiente virtual ===="

python -m venv env

echo "==== Ativando o ambiente virtual ===="


source env/bin/activate

echo "==== Preparando a instalação de pacotes ===="


sudo apt-get remove -y python-blinker || true
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==== Configurando o Ollama ===="

curl -fsSL https://ollama.com/install.sh | sh

sleep 5

nohup ollama serve &

sleep 2

ollama pull llama3.2
ollama pull deepseek-r1:latest
ollama pull mxbai-embed-large:latest

sleep 2



clear

echo "==== Iniciando aplicação ===="
nohup waitress-serve --listen=0.0.0.0:8000 main:app &