#!/bin/bash

clear

echo "==== Incializando ===="

apt update

echo "==== repositórios do linux atualizados ====="

apt install -y vim

echo "==== Editor de texto instalado: VIM ====="

echo "Iniciando a configuração e instalação do ambiente de execução Em:"


sleep 1

echo "3"

sleep 1

echo "2"

sleep 1

echo "1"

sleep 1

clear


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

echo "Llama3.2:"
ollama pull llama3.2

echo "Deepseek-r1"
ollama pull deepseek-r1:latest

echo "mxbai-embed-large"
ollama pull mxbai-embed-large:latest

echo "gpt-oss:latest"
ollama pull gpt-oss:latest

sleep 2

clear

ollama ls

sleep 5

clear

echo "==== Iniciando aplicação ===="
# nohup waitress-serve --listen=0.0.0.0:8000 main_router:app &