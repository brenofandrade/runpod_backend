# main.py
# Para servir a aplicação com waitress:
# waitress-serve --listen=0.0.0.0:8000 main:app

import os
import logging
from warnings import filterwarnings
from typing import List, Dict, Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---- Config base ----
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

# Falha cedo se variáveis críticas faltarem
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME não configurada.")

# ---- Funções utilitárias ----
def serialize_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    out = []
    for i, d in enumerate(docs or []):
        out.append(
            {
                "rank": i + 1,
                "text": (d.page_content or "")[:500],
                "metadata": d.metadata or {},
            }
        )
    return out

# ---- App ----
app = Flask(__name__)
CORS(app)

# ---- Recursos externos (Pinecone / Ollama / LangChain) ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

llm = ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Instanciamos o VectorStore sem travar namespace; passaremos namespace no retriever.
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

SYSTEM_PROMPT = (
    "Você é um assistente RAG. Responda de forma concisa usando APENAS os trechos do contexto. "
    "Se não souber, diga 'Não sei com base nos documentos'.\n\n"
    "# Contexto\n{context}\n\n# Pergunta\n{input}"
)

# Importante: o prompt precisa ter {context} e {input} (não {question}).
prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
combine_chain = create_stuff_documents_chain(llm, prompt)
# O retriever será criado por requisição (pra suportar k/namespace dinâmicos).

# ---- Rotas ----
@app.get("/health")
def health():
    try:
        # Verificação leve de conectividade com Pinecone
        _ = index.describe_index_stats()
        # Verificação leve de Ollama (chamada mínima via geração “ping”)
        _ = llm.invoke("pong?")  # sem contexto, só sanity check
        return jsonify({"status": "OK"}), 200
    except Exception as e:
        logging.exception("Health check falhou: %s", e)
        return jsonify({"status": "DEGRADED", "error": str(e)}), 503

@app.get("/upvecstore")
def update_vector_store():
    pass


@app.post("/chat")
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    question: str = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400

    # parâmetros opcionais
    k = int(payload.get("k") or 4)
    namespace = (payload.get("namespace") or DEFAULT_NAMESPACE).strip()

    logging.info(f"Question: {question} | k={k} | namespace={namespace!r}")

    try:
        # retriever por request (pra aceitar k/namespace)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "namespace": namespace},
        )

        rag_chain = create_retrieval_chain(retriever, combine_chain)

        # Chave correta para o chain: "input"
        result = rag_chain.invoke({"input": question})

        answer = result.get("answer", "").strip()
        context_docs = result.get("context", [])

        response = {
            "answer": answer,
            "sources": serialize_sources(context_docs),
            "namespace": namespace,
            "k": k,
        }
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Erro no chat: %s", e)
        return jsonify({"error": str(e)}), 500



# ---- Execução direta (dev) ----
if __name__ == "__main__":
    # Em produção prefira waitress-serve (ver cabeçalho).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
