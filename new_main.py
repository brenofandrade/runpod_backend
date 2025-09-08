# main.py
# Para rodar: waitress-serve --listen=0.0.0.0:8000 main:app

# --- Imports e Configuração Base ---
import os
import json
import logging
import re
from warnings import filterwarnings
from typing import List, Dict, Any

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.globals import set_debug, set_verbose

from pinecone import Pinecone
from pydantic import BaseModel, Field


# --- Ambiente e Logging ---
load_dotenv(override=True)
set_debug(True)
set_verbose(True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
filterwarnings("ignore")


# --- Variáveis de Ambiente ---
OLLAMA_BASE_URL     = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL    = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL     = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DEFAULT_NAMESPACE   = os.getenv("PINECONE_NAMESPACE", "default")

# --- Validação de Variáveis Críticas ---
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME não configurada.")


# --- Prompt e Parsing do Roteador ---
ROUTER_SYSTEM = """
Você é um roteador de perguntas para um sistema RAG.
Responda apenas com JSON com os campos:
- "use_rag"
- "confidence"
- "rationale"

Exemplo de resposta válida:
{{"use_rag": true, "confidence": 0.87, "rationale": "Cita contrato interno"}}
"""




# """
# Você é um roteador de perguntas para um sistema RAG.
# Decida se a pergunta precisa consultar a base (RAG) ou se pode ser respondida sem RAG.
# Responda apenas com JSON válido: {"use_rag": true, "confidence": 0.85, "rationale": "motivo"}
# """

class RouteDecision(BaseModel):
    use_rag: bool = Field(..., description="Se deve usar RAG.")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da decisão.")
    rationale: str = Field(..., description="Justificativa curta (pt-br)")

router_llm = ChatOllama(model="llama3.2", temperature=0, base_url=OLLAMA_BASE_URL)
router_parser = JsonOutputParser(pydantic_object=RouteDecision)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("user", "Pergunta: {question}")
])


# --- Cadeias LLM ---
direct_llm = ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente técnico. Responda em português. Seja preciso."),
    ("user", "{question}")
])
direct_chain = direct_prompt | direct_llm


# --- Funções Utilitárias ---
memory_store = {}

def get_memory(session_id):
    return memory_store.setdefault(session_id, [])

def _strip_fences_and_think(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    s = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()

def _extract_first_json_object(s: str) -> str:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else ""

def _looks_like_json_schema(d: dict) -> bool:
    return isinstance(d, dict) and (
        ("properties" in d and "type" in d) or
        set(d.keys()) <= {"type", "properties", "required", "title", "$schema"}
    )

def serialize_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "text": (doc.page_content or "")[:500],
            "metadata": doc.metadata or {},
        }
        for i, doc in enumerate(docs or [])
    ]

def llm_route(question: str) -> RouteDecision:
    msg = router_prompt.invoke({"question": question})
    out = router_llm.invoke(msg)

    raw = getattr(out, "content", out)
    if not isinstance(raw, str):
        raw = str(raw)

    logging.debug(f"[Router LLM] bruto: {raw!r}")
    cleaned = _strip_fences_and_think(raw)
    candidate = _extract_first_json_object(cleaned) or cleaned

    # Tenta parse com LangChain
    try:
        parsed = router_parser.parse(candidate)
        if isinstance(parsed, RouteDecision):
            return parsed
    except Exception as e:
        logging.warning(f"[Router Parser] Falhou, tentando json.loads. Erro: {e}")

    # Fallback manual
    try:
        parsed_json = json.loads(candidate)
        if _looks_like_json_schema(parsed_json):
            raise ValueError("Router devolveu um JSON Schema, não um objeto de decisão.")

        return RouteDecision(**parsed_json)

    except Exception as e:
        logging.error(f"[Router] JSON inválido para decisão: {candidate!r}")
        raise ValueError("Não foi possível interpretar a decisão do roteador.") from e



# --- Conexões com Pinecone e LangChain ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

prompt_rag = ChatPromptTemplate.from_template("""
Você é um assistente especializado. Responda apenas com base no contexto.
{context}

Pergunta: {input}
Resposta:
""".strip())

combine_chain = create_stuff_documents_chain(direct_llm, prompt_rag)


# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)
app.url_map.strict_slashes = False


# --- Endpoints ---
@app.get("/health")
def health_check():
    try:
        index.describe_index_stats()
        return jsonify({"status": "OK"}), 200
    except Exception as e:
        logging.exception("Health check falhou.")
        return jsonify({"status": "DEGRADED", "error": str(e)}), 503


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    
    session_id = request.headers.get("X-Session-Id")
    
    
    if not session_id:
        return jsonify({"error": "Faltando session_id"}), 400
    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400

    memory = ConversationBufferMemory(memory_key="history", return_messages=True)
    memory.chat_memory.messages = get_memory(session_id)

    # Params opcionais
    k = int(payload.get("k") or 4)
    namespace = (payload.get("namespace") or DEFAULT_NAMESPACE).strip()
    force_rag = bool(payload.get("force_rag") or False)
    force_direct = bool(payload.get("force_direct") or False)

    if force_rag and force_direct:
        return jsonify({"error": "force_rag e force_direct não podem ser usados juntos."}), 400

    logging.info(f"Pergunta: {question} | k={k} | ns={namespace} | force_rag={force_rag} | force_direct={force_direct}")

    try:
        use_rag = force_rag if (force_rag or force_direct) else llm_route(question).use_rag

        if not use_rag:
            direct_response = direct_chain.invoke({"question": question, "history" : memory.chat_memory.messages})
            resposta = _strip_fences_and_think(getattr(direct_response, "content", "Sem resposta."))

            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(resposta)
            memory_store[session_id] = memory.chat_memory.messages

            return jsonify({
                "answer": resposta,
                "sources": [],
                "namespace": None,
                "k": 0,
                "user_rag": False,
            }), 200

        # RAG Flow
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "namespace": namespace}
        )
        rag_chain = create_retrieval_chain(retriever, combine_chain)
        result = rag_chain.invoke({"input": question})

        answer = _strip_fences_and_think(result.get("answer", ""))
        context_docs = result.get("context", [])

        if not context_docs:
            logging.info("[RAG] Sem contexto. Fallback para resposta direta.")
            fallback = direct_chain.invoke({"question": question})
            answer = _strip_fences_and_think(getattr(fallback, "content", "Não sei."))

        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        memory_store[session_id] = memory.chat_memory.messages

        return jsonify({
            "answer": answer,
            "sources": serialize_sources(context_docs),
            "namespace": namespace,
            "k": k,
            "used_rag": bool(context_docs),
        }), 200

    except Exception as error:
        logging.exception("Erro no /chat")
        return jsonify({"error": str(error)}), 500


# --- Execução Local ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
