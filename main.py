# ////////////////////////////////////////////////////////////////////////////////////////
#
# Chat_RAG System
#
# Unimed Blumenau
# 
# Criado por Breno Andrade
#
# Descrição: API que serve o LLM para interação de chat
#
# Estrutura:    1) Imports
#               2) Inicia as variaveis de ambiente
#               3) Helper Functions
#               4) Rotas
#                   a) Check backend health
#                   b) Call backend route for /chat
#
# Para servir a aplicação com waitress:
# waitress-serve --listen=0.0.0.0:8000 main:app
#
# ////////////////////////////////////////////////////////////////////////////////////////

# --- Imports e Configuração Base ---
import os
import re
import json
import time

import numpy as np

import random
import logging
from warnings import filterwarnings
from typing import List, Dict, Any, Tuple
from collections import OrderedDict
from threading import RLock
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.globals import set_debug, set_verbose
from sentence_transformers import CrossEncoder


# --- Ambiente e Logging ---
load_dotenv(override=True)
filterwarnings("ignore")

# Para debugar o código em execução
set_debug(True)
set_verbose(True)

# --- Variáveis de Ambiente ---
LOG_LEVEL             = os.getenv("LOG_LEVEL", "INFO").upper()
OLLAMA_BASE_URL       = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL      = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME   = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
DEFAULT_NAMESPACE     = os.getenv("PINECONE_NAMESPACE", "default")
RETRIEVAL_K           = int(os.getenv("RETRIEVAL_K", "2"))
OPENAI_KEY            = os.getenv("OPENAI_API_KEY")

RERANK_METHOD_DEFAULT = os.getenv("RERANK_METHOD_DEFAULT", "none").lower()
RERANK_TOP_K_DEFAULT  = int(os.getenv("RERANK_TOP_K_DEFAULT", "0"))  # 0 or empty means “use top_k if not specified”
CROSS_ENCODER_MODEL   = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_BATCH_SIZE     = int(os.getenv("RERANK_BATCH_SIZE", "16"))

MAX_HISTORY           = int(os.getenv("MAX_HISTORY", "10"))
TTL_SETUP             = int(os.getenv("TTL_SETUP", "1200"))


model_name = CROSS_ENCODER_MODEL

CROSS_ENCODER_CACHE = None
def get_cross_encoder(model_name):
    global CROSS_ENCODER_CACHE
    if CROSS_ENCODER_CACHE is None:
        CROSS_ENCODER_CACHE = CrossEncoder(model_name)
    return CROSS_ENCODER_CACHE

cross_model = get_cross_encoder(model_name)



logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Validação de Variáveis Críticas ---
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX (ou PINECONE_INDEX_NAME) não configurada.")


# --- OpenAI (opcional para gerar variações de consulta) ---
client = None
if OPENAI_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
    except Exception as e:
        logger.warning("Falha ao inicializar OpenAI: %s", e)
        client = None
else:
    logger.info("OPENAI_API_KEY não configurada; usarei fallback para variações de consulta.")

# --- Pinecone / Embeddings / VectorStore ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- LLM para geração ---
llm = ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Cria a conexão com o Vectorstore
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# =========================
# MÓDULO DE MEMÓRIA EM RAM
# =========================
# Estrutura:
# memory_store = {
#   session_id: { "messages": [..], "expires_at": epoch_float }
# }
memory_store: Dict[str, Dict[str, Any]] = {}
_memory_lock = RLock()

def _now() -> float:
    return time.time()

def _is_expired(entry: Dict[str, Any]) -> bool:
    return bool(entry) and entry.get("expires_at", 0) <= _now()

def _ensure_entry(session_id: str, ttl: int = TTL_SETUP) -> Dict[str, Any]:
    with _memory_lock:
        entry = memory_store.get(session_id)
        if not entry or _is_expired(entry):
            memory_store[session_id] = {"messages": [], "expires_at": _now() + ttl}
        return memory_store[session_id]

def get_memory(session_id: str) -> List[Any]:
    """Retorna a lista de mensagens da sessão (cria se não existir/expirada)."""
    entry = _ensure_entry(session_id)
    return entry["messages"]

def clear_memory(session_id: str) -> bool:
    """Remove a sessão da memória RAM."""
    with _memory_lock:
        return memory_store.pop(session_id, None) is not None

def save_memory(session_id: str, messages: List[Any], ttl: int = TTL_SETUP) -> None:
    """Salva mensagens e renova o TTL."""
    with _memory_lock:
        memory_store[session_id] = {
            "messages": list(messages),
            "expires_at": _now() + ttl,
        }

def load_memory(session_id: str) -> List[Any]:
    """Carrega mensagens respeitando TTL; se expirado, limpa e retorna lista vazia."""
    with _memory_lock:
        entry = memory_store.get(session_id)
        if not entry:
            return []
        if _is_expired(entry):
            memory_store.pop(session_id, None)
            return []
        return list(entry.get("messages", []))

def update_memory(session_id: str, messages: List[Any]) -> List[Any]:
    """Trunca pelo MAX_HISTORY, salva e retorna o recorte."""
    truncated = list(messages[-MAX_HISTORY:])
    save_memory(session_id, truncated, ttl=TTL_SETUP)
    return truncated
# =========================

# --- Prompts ---
prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é o Assistente interno da Unimed para dúvidas de colaboradores.  
            Responda usando SOMENTE o conteúdo em "Contexto".  
            Se houver trechos relacionados, mesmo que parciais, utilize-os para formular uma resposta útil.  
            Evite inventar informações factuais (valores, percentuais, versões de documentos, missão, visão, benefícios) que não estejam no Contexto.

            ## Objetivo
            Fornecer respostas úteis, claras e confiáveis para apoiar o trabalho do dia a dia, com foco prático.

            ## Diretrizes
            - Utilize as evidências do Contexto da melhor forma possível, mesmo que incompletas.  
            - Use fallback somente quando realmente não existir nenhum ponto relevante no Contexto.  
            - Se a informação for parcial, explique de forma clara o que está presente e o que está faltando.  
            - Se houver divergência, informe a inconsistência e recomende validação com a área responsável.  
            - Escreva em português do Brasil, com linguagem profissional, cordial e objetiva.  
            - Traga passo a passo **apenas** quando a pergunta indicar um procedimento ou ação.  
            - Não inclua seção “Fontes” nem nomes/códigos de documentos.  
            - Evite repetir a pergunta do usuário.

            ## Mensagens de fallback
            Se não houver conteúdo aplicável no Contexto, use UMA das mensagens:  
            - "Não localizei informação suficiente no Contexto para responder com segurança. Se possível, reformule a pergunta incluindo o sistema, processo ou área envolvidos."  
            - "O Contexto traz menções relacionadas, mas sem detalhes suficientes para orientar com clareza. Recomendo validar com as áreas ou sistemas citados."  
            - "O Contexto apresenta informações divergentes sobre este tema. Para evitar erro, valide com o setor responsável e confira a versão mais recente disponível."

            ## Organização da resposta
            Use parágrafos claros e objetivos.  
            Quando fizer sentido, estruture assim:
            - **Resumo** (1–3 frases, apenas para perguntas longas/complexas)  
            - **Conteúdo principal**  
            - **Passo a passo** (apenas se a pergunta pedir instruções)  
            - **Observações/Regras**  

            Inclua trechos do Contexto em citações curtas com Markdown (`>`), mas nunca mencione arquivos ou páginas.
            """
        ),
        MessagesPlaceholder(variable_name="history"),
        ("system", "# Contexto:\n{context}"),
        ("user", "# Pergunta:\n{input}\n\n# Resposta:")
    ]
)

prompt_general = ChatPromptTemplate.from_messages(
    [
        ("system",
         "Você é um assistente útil e responsável. "
         "Responda de forma geral e educativa, sem inventar fatos sobre a empresa. "
         "Nunca forneça (nem peça) dados pessoais, credenciais, segredos ou informações confidenciais. "
         "Não dê aconselhamento jurídico/financeiro específico. Se algo for especulativo, diga que não sabe."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ]
)


# Helpers
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

def extract_gen_usage(ai_msg: AIMessage) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    gen = {}
    usage = {}
    if hasattr(ai_msg, "response_metadata") and ai_msg.response_metadata:
        gen = {**gen, **(ai_msg.response_metadata or {})}
    if hasattr(ai_msg, "generation_info") and getattr(ai_msg, "generation_info"):
        gen = {**gen, **(ai_msg.generation_info or {})}
    if hasattr(ai_msg, "usage_metadata") and ai_msg.usage_metadata:
        usage = {**usage, **(ai_msg.usage_metadata or {})}
    add = getattr(ai_msg, "additional_kwargs", {}) or {}
    if not gen:
        gen = add.get("response_metadata") or add.get("generation_info") or {}
    if not usage:
        usage = add.get("usage_metadata") or add.get("token_usage") or add.get("usage") or {}
    return gen, usage

def _fallback_variants(question: str, n: int) -> List[str]:
    q = question.strip()
    base = [q]
    extras = [q.lower(), q.upper()]
    out = []
    for s in base + extras:
        s = s.strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out

def generate_llm_variants(question: str, n: int = 4) -> List[str]:
    if not question or not question.strip():
        return []
    if client is None:
        return _fallback_variants(question, n)

    pool_prompt = (
        "Inclua palavras como 'como', 'procedimento', 'área/unidade', 'ação/processo', 'quem', 'onde solicitar', 'quais critérios'. "
        "(ex.: citar área/unidade responsável, documentos internos, públicos específicos, tipos de colaborador). "
    )

    prompt = (
        f"Gere {n} variações curtas e diferentes da pergunta abaixo."
        "Use sinônimos e termos próximos, mas também expanda a pergunta com contextos prováveis "
        "As variações devem ser mais específicas que a pergunta original, sempre mantendo a intenção central. "
        "Se aplicável, substitua termos genéricos por sinônimos de glossário (ex.: treinamento = capacitação, curso, T&D). "
        "Se o termo for abreviação, substitua pelo nome por extenso."
        "Uma por linha, sem numeração.\n\n"
        f"Pergunta: {question}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "Você ajuda a reescrever perguntas de modo útil para busca."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content or ""
        variants = [line.strip("•- \t") for line in content.splitlines() if line.strip()]
        if question.strip() not in variants:
            variants.insert(0, question.strip())
        return variants[:n]
    except Exception as e:
        logger.warning("Erro ao gerar variações com OpenAI: %s", e)
        return _fallback_variants(question, n)

def rerank_by_embedding(query:str, docs:List[Document], embed_function, top_k:int) -> List[Document]:
    """Re-rank documentos pelo método do cosseno da similaridade entre vetores que representa a query e os documentos. Mesmo usado na busca"""

    if not docs:
        return docs

    query_vector = np.array(embed_function.embed_query(query), dtype=float)

    doc_texts = [d.page_content or "" for d in docs]
    doc_vector = embed_function.embed_documents(doc_texts)
    doc_vector = [np.array(vec, dtype=float) for vec in doc_vector]

    if np.linalg.norm(query_vector) != 0:
        query_vector = query_vector / np.linalg.norm(query_vector)

    docs_normalizados = []
    for vec in doc_vector:
        if np.linalg.norm(vec) != 0:
            docs_normalizados.append(vec / np.linalg.norm(vec))
        else:
            docs_normalizados.append(vec)

    scores = [float(np.dot(query_vector, doc_vec)) for doc_vec in docs_normalizados]
    scored_docs = list(zip(scores, range(len(docs)), docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    top_scored = scored_docs[:top_k]

    reranked_docs = []
    for score, _, doc in top_scored:
        doc.metadata['rerank_score'] = round(score, 6)
        doc.metadata['rerank_method'] = "embedding"
        reranked_docs.append(doc)
    return reranked_docs


def rerank_by_cross_encoder(
        query:str,
        docs:List[Document],
        model_name:str,
        embed_function, 
        top_k:int,
        batch_size:int
) -> List[Document]:
    """Re-rank documentos usando modelo CrossEncoder para par query-document"""

    if not docs:
        return docs
    
    try:
        cross_model = get_cross_encoder(model_name)
    except Exception as e:
        return rerank_by_embedding(query, docs, embed_function, top_k)

    sentences = [(query, d.page_content) for d in docs]
    scores = cross_model.predict(sentences, batch_size=batch_size)

    # Sort docs by score
    scored_docs = list(zip(scores, range(len(docs)), docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_scored = scored_docs[:top_k]

    # Attach score and method to metadata
    reranked_docs = []
    for score, _, doc in top_scored:
        doc.metadata["rerank_score"] = float(score)
        doc.metadata["rerank_method"] = "cross_encoder"
        reranked_docs.append(doc)
    
    return reranked_docs

def retrieve_union(
        queries: List[str], 
        k: int, 
        namespace: str,
        rerank_method: str = "None",
        rerank_top_k: int = None
        ) -> List[Document]:
    """
    Recuperação de documentos para multi-consulta, deduplicação e opcionalmente re-rank.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "namespace": namespace},
    )

    collected: List[Document] = []
    seen = set()
    for q in queries:
        try:
            docs = retriever.get_relevant_documents(q)
        except Exception as e:
            logger.error("Erro no retriever para '%s': %s", q, e)
            continue
        for d in docs or []:
            key = hash((d.page_content or "").strip())
            if key not in seen:
                seen.add(key)
                collected.append(d)

    # Re-rank docs para geração com mais relevância
    rerank_method = (rerank_method or "none").lower()
    if rerank_method == "none":
        final_top_k = rerank_top_k if rerank_top_k else k
        return collected[:final_top_k] if final_top_k and len(collected) > final_top_k else collected
    elif rerank_method == "embedding":
        reranked = rerank_by_embedding(queries[0] if queries else "", collected, embeddings, rerank_top_k)
        logger.debug(f"Embedding re-rank: {len(collected)} coletados, retornou top {len(reranked)}.")
        if reranked:
            top_scores = [doc.metadata.get("rerank_score") for doc in reranked[:3]]
            logger.debug(f"Top-3 Embedding scores: {top_scores}")
        return reranked
    elif rerank_method == "cross-encoder":
        reranked = rerank_by_cross_encoder(queries[0] if queries else "", collected, CROSS_ENCODER_MODEL, embeddings, rerank_top_k, RERANK_BATCH_SIZE)
        logger.debug(f"Cross-encoder re-rank: considered {len(collected)} coletados, retornou top {len(reranked)}.")
        if reranked:
            top_scores = [doc.metadata.get('rerank_score') for doc in reranked[:3]]
            logger.debug(f"Top-3 cross-encoder scores: {top_scores}")
        return reranked
    else:
        return collected[:rerank_top_k] if rerank_top_k else collected

def serialize_sources(docs: List[Document], max_chars: int = 900) -> List[Dict[str, Any]]:
    out = []
    for i, doc in enumerate(docs or []):
        meta = doc.metadata or {}
        ident = (
            meta.get("source") or meta.get("file_path") or meta.get("filename") or
            meta.get("document_id") or meta.get("doc_id") or ""
        )
        page = meta.get("page") or meta.get("page_number") or meta.get("loc", {}).get("page") or None
        text = (doc.page_content or "").strip()
        out.append({
            "rank": i + 1,
            "id": ident,
            "page": page,
            "text_preview": text[:max_chars],
            "text_len": len(text),
            "metadata": meta
        })
    return out

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join((d.page_content or "") for d in docs)

def _to_bool(val, default=False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"1","true","t","yes","y","on"}:
            return True
        if s in {"0","false","f","no","n","off"}:
            return False
    return default

# --- Flask App ---
app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "OPTIONS"],
    max_age=86400,
)
app.url_map.strict_slashes = False

@app.route("/health", methods=["GET"])
def health():
    
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.perf_counter()
    try:
        payload = request.get_json(force=True, silent=False) or {}
        question = (payload.get("question") or "").strip()
        session_id = request.headers.get("X-Session-Id", "")
        rerank_method = (payload.get("rerank_method") or RERANK_METHOD_DEFAULT).strip().lower()
        rerank_top_k = payload.get("rerank_top_k", None)
        if rerank_top_k is None:
            rerank_top_k = RERANK_TOP_K_DEFAULT
        
        raw_use_rag = payload.get("use_rag", payload.get("force_rag", True))
        use_rag = _to_bool(raw_use_rag, default=True)

        if not session_id:
            return jsonify({"error": "Faltando session id"}), 400
        if not question:
            return jsonify({"error": "Campo 'question' é obrigatório."}), 400

        # memória (RAM)
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        memory.chat_memory.messages = load_memory(session_id)

        if not use_rag:

            ai_msg:AIMessage=(prompt_general|llm).invoke({
                "input":question,
                "history":memory.chat_memory.messages
            })
            answer = _strip_fences_and_think(ai_msg.content)

            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            update_memory(session_id, memory.chat_memory.messages)
            
            latency = (time.perf_counter() - start_time)*1000

            logger.info(f"Latência /chat (no-RAG) = {latency:.2f} ms")

            
            return jsonify({
                "question_original": question,
                "answer": answer,
                "latency_ms": latency,
                "mode": "no_rag",
                "sources":None
            }), 200

        # Parâmetros opcionais
        k = int(payload.get("k") or RETRIEVAL_K)
        namespace = (payload.get("namespace") or DEFAULT_NAMESPACE or "default").strip()

        # 1) Gera variações de consulta
        multi_query = generate_llm_variants(question, n=2)
        generate_question = random.choice(multi_query[1:]) if len(multi_query) > 1 else multi_query[0]

        # 2) Recupera contexto (união deduplicada)
        context_docs: List[Document] = retrieve_union(
            queries=multi_query, 
            k=k, 
            namespace=namespace, 
            rerank_method=rerank_method,
            rerank_top_k=rerank_top_k
            )
        context_text = format_docs(context_docs)

        # 3) Gera resposta
        ai_msg: AIMessage = (prompt_rag | llm).invoke({
            "input": question,
            "history": memory.chat_memory.messages,
            "context": context_text
        })

        answer = _strip_fences_and_think(ai_msg.content)
        gen_info, usage_info = extract_gen_usage(ai_msg)

        # Atualiza memória (RAM)
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        update_memory(session_id, memory.chat_memory.messages)

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Latência /chat = {latency:.2f} ms")
        

        return jsonify({
            "question_original": question,
            "question_used": generate_question,
            "answer": answer,
            "latency_ms": latency,
            "mode": "rag",
            "sources": serialize_sources(context_docs),
            "retrieval": {
                "k": k,
                "namespace": namespace,
                "variants": multi_query,
                "docs": len(context_docs),
            },
            "metadata": {
                "generation_info": gen_info,
                "usage_info": usage_info
            }
        }), 200

    except Exception as e:
        logger.exception("Erro no endpoint /chat")
        return jsonify({"error": "Erro interno ao processar a solicitação.", "detail": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    try:
        
        session_id = request.headers.get("X-Session-Id")

        if not session_id:
            payload = request.get_json(silent=True) or {}
            session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "Informe o session_id no header X-Session-Id ou no corpo JSON."}), 400
            

        removed = clear_memory(session_id)

        if removed:
            return jsonify({"status": "ok", "message": f"Memória da sessão {session_id} foi limpa."}), 200
        else:
            return jsonify({"status": "not_found", "message": f"Nenhuma memória encontrada para {session_id}."}), 404

    except Exception as e:
        logger.exception("Erro no endpoint /clear")
        return jsonify({"error": "Erro interno ao limpar memória.", "detail": str(e)}), 500




# ---- Execução  ----
if __name__ == "__main__":
    # Em produção prefira waitress-serve (ver cabeçalho).
    # app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
    app.run(host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")