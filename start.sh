#!/bin/bash



ollama pull llama3.2
ollama pull mxbai-embed-large


sleep 2


echo "==== Criando ambiente virtual ===="
python -m venv .venv && source .venv/bin/activate

sudo apt-get remove -y python-blinker || true
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "==== Iniciando aplicação ===="
python app.py