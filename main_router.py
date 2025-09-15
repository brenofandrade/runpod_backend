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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser  # (não usado, mas útil caso queira experimentar)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.globals import set_debug, set_verbose
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore

from pydantic import BaseModel, Field

# --- Ambiente e Logging ---
load_dotenv(override=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
filterwarnings("ignore")

# --- Variáveis de Ambiente ---
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL  = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
# CORREÇÃO: pegue PINECONE_API_KEY corretamente, sem defaults enganosos
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")
# debug para visualizar o fluxo do langchain
set_debug(True) 
set_verbose(True)
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

Exemplo de resposta válida:
{"use_rag": true, "confidence": 0.87}
"""

class RouteDecision(BaseModel):
    use_rag: bool = Field(..., description="Se deve usar RAG.")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da decisão.")

# Roteador simples usando o próprio Ollama (sem parser do LangChain para não perder metadados)
router_llm = ChatOllama(model="llama3.2", temperature=0, base_url=OLLAMA_BASE_URL)
router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_SYSTEM),
    ("user", "Pergunta: {question}")
])

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

def llm_route(question: str) -> RouteDecision:
    msg = router_prompt.invoke({"question": question})
    out: AIMessage = router_llm.invoke(msg)
    raw = getattr(out, "content", "")
    cleaned = _strip_fences_and_think(raw)
    candidate = _extract_first_json_object(cleaned) or cleaned

    try:
        parsed_json = json.loads(candidate)
        if _looks_like_json_schema(parsed_json):
            raise ValueError("Router devolveu um JSON Schema, não um objeto de decisão.")
        return RouteDecision(**parsed_json)
    except Exception as e:
        logging.error(f"[Router] JSON inválido: {candidate!r}")
        # fallback conservador
        return RouteDecision(use_rag=True, confidence=0.5)

# --- Cadeias LLM (direto e RAG) ---
direct_llm = ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
direct_prompt = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente técnico. Responda em português. Seja preciso."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])

# Prompt do RAG preservando metadados (não usar create_stuff_documents_chain aqui)
prompt_rag = ChatPromptTemplate.from_messages([
    ("system", "Você é um assistente especializado em responder **apenas** com base no Contexto."),
    MessagesPlaceholder(variable_name="history"),
    ("system",
     "Regras:\n"
     "- Use exclusivamente o conteúdo em “Contexto”.\n"
     "- Seja conciso (<=120 palavras, exceto código/tabelas).\n"
     "- Se não souber, diga: \"Não encontrei uma referência interna confiável para responder com segurança.\"\n"
     "- Cite fontes no formato [fonte:source|doc_id|p.page] quando disponíveis.\n"
     "- Se houver conflito no contexto, informe o conflito sem extrapolar.\n"
     "- Responda em português."),
    ("system", "# Contexto:\n{context}"),
    ("user", "# Pergunta:\n{input}\n\n# Resposta:")
])

# --- Funções Utilitárias ---
memory_store: Dict[str, List] = {}

def get_memory(session_id):
    return memory_store.setdefault(session_id, [])

def clear_memory(session_id: str) -> bool:
    return memory_store.pop(session_id, None) is not None

def serialize_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "text": (doc.page_content or "")[:500],
            "metadata": doc.metadata or {},
        }
        for i, doc in enumerate(docs or [])
    ]

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join((d.page_content or "") for d in docs)

def extract_gen_usage(ai_msg: AIMessage) -> (Dict[str, Any], Dict[str, Any]):
    """
    Extrai de forma robusta metadados de geração e uso.
    Funciona com LangChain 0.2/0.3+ e provedores via Ollama.
    """
    gen = {}
    usage = {}

    # Preferência: atributos dedicados
    if hasattr(ai_msg, "response_metadata") and ai_msg.response_metadata:
        gen = {**gen, **(ai_msg.response_metadata or {})}
    if hasattr(ai_msg, "generation_info") and getattr(ai_msg, "generation_info"):
        gen = {**gen, **(ai_msg.generation_info or {})}
    if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
        usage = {**usage, **(ai_msg.usage_metadata or {})}

    # Fallback: additional_kwargs
    add = getattr(ai_msg, "additional_kwargs", {}) or {}
    if not gen:
        gen = add.get("response_metadata") or add.get("generation_info") or {}
    if not usage:
        usage = add.get("usage_metadata") or add.get("token_usage") or add.get("usage") or {}

    return gen, usage

# --- Conexões com Pinecone e VectorStore ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

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

    # Memória
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
        if (force_rag or force_direct):
            use_rag = True 
        else:
            response_route = llm_route(question)
            use_rag = response_route.get("use_rag")

        # ------------------ Fluxo Direto ------------------
        if not use_rag:
            ai_msg: AIMessage = (direct_prompt | direct_llm).invoke(
                {"question": question, "history": memory.chat_memory.messages}
            )
            resposta = _strip_fences_and_think(ai_msg.content)
            gen_info, usage_info = extract_gen_usage(ai_msg)

            # Atualiza memória
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(resposta)
            memory_store[session_id] = memory.chat_memory.messages

            return jsonify({
                "answer": resposta,
                "sources": [],
                "namespace": None,
                "k": 0,
                "used_rag": False,
                "metadata": {
                    "generation_info": gen_info,
                    "usage_info": usage_info
                }
            }), 200

        # ------------------ Fluxo RAG (preservando metadados) ------------------
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "namespace": namespace}
        )
        context_docs: List[Document] = retriever.invoke(question) or []
        context_text = format_docs(context_docs)

        rag_ai: AIMessage = (prompt_rag | direct_llm).invoke({
            "input": question,
            "history": memory.chat_memory.messages,
            "context": context_text
        })

        answer = _strip_fences_and_think(rag_ai.content)
        gen_info, usage_info = extract_gen_usage(rag_ai)

        FALLBACK_SENTENCE = "Não encontrei uma referência interna confiável para responder com segurança."

        used_rag = bool(context_docs) and (answer != FALLBACK_SENTENCE)

        # Fallback para resposta direta se não houver contexto útil
        if not used_rag:
            logging.info("[RAG] Sem contexto útil. Fallback para resposta direta.")
            ai_msg_fb: AIMessage = (direct_prompt | direct_llm).invoke(
                {"question": question, "history": memory.chat_memory.messages}
            )
            answer = _strip_fences_and_think(ai_msg_fb.content)
            gen_info, usage_info = extract_gen_usage(ai_msg_fb)
            context_docs = []
            namespace_return = None
            k_return = 0
        else:
            namespace_return = namespace
            k_return = k

        # Atualiza memória
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        memory_store[session_id] = memory.chat_memory.messages

        return jsonify({
            "answer": answer,
            "sources": serialize_sources(context_docs),
            "namespace": namespace_return,
            "k": k_return,
            "used_rag": used_rag,
            "metadata": {
                "generation_info": gen_info,
                "usage_info": usage_info
            }
        }), 200

    except Exception as error:
        logging.exception("Erro no /chat")
        return jsonify({"error": str(error)}), 500

@app.post("/reset")
def reset_conversation():
    payload = request.get_json(force=True, silent=True) or {}
    session_id = request.headers.get("X-Session-Id") or payload.get("session_id")

    if not session_id:
        return jsonify({"error": "Faltando session_id"}), 400

    cleared = clear_memory(session_id)

    return jsonify({
        "ok": True,
        "cleared": cleared,
        "session_id": session_id,
        "message": "Memória apagada. Nova conversa iniciada."
    }), 200

# --- Execução Local ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=False)
