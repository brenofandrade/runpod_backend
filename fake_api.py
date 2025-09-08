

import os
import logging
from warnings import filterwarnings
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from pinecone import Pinecone # Depreacated
from langchain.globals import set_debug, set_verbose
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import datetime


set_debug(True)
set_verbose(True)
load_dotenv(override=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
filterwarnings("ignore")



OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL    = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-vector-store")
DEFAULT_NAMESPACE   = os.getenv("PINECONE_NAMESPACE", "default")


app = Flask(__name__)
# CORS amplo e com suporte a preflight
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)


@app.get("/health")
def health():
    try:
        print("[OK] - Chamada recebida")
        return jsonify({"status": "OK"}), 200
    except Exception as e:
        logging.exception("Health check falhou: %s", e)
        return jsonify({"status": "DEGRADED", "error": str(e)}), 503
    
@app.route("/chat", methods=["POST", "OPTIONS"])
@app.route("/chat/", methods=["POST", "OPTIONS"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    question:str=(payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400
    
    try:
        resultado = f"A resposta para a pergunta: '{question}' é 42"
        
        print(resultado)
        
        response = {
            "answer":resultado,
            "source":"inexistente",
            "timestamp":datetime.now()
        }

        return jsonify(response), 200
    except Exception as error:
        return jsonify({"error":str(error)}), 500
    
if __name__ == "__main__":
    # Em produção prefira waitress-serve (ver cabeçalho).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)