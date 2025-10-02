# Para rodar: waitress-serve --listen=0.0.0.0:8000 main_rag:app
# --- Imports e Configuração Base ---
import os
import re
import json
import time
import redis
import random
import logging
from warnings import filterwarnings
from typing import List, Dict, Any
from collections import OrderedDict
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
# from pydantic import BaseModel, Field

# --- Ambiente e Logging ---
load_dotenv(override=True)
filterwarnings("ignore")
set_debug(True) 
set_verbose(True)

# --- Variáveis de Ambiente ---
LOG_LEVEL            = os.getenv("LOG_LEVEL", "INFO").upper()
OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL     = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL      = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
# PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY_DSUNIBLU")
# PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX") or os.getenv("PINECONE_INDEX_NAME")
DEFAULT_NAMESPACE    = os.getenv("PINECONE_NAMESPACE", "default")
RETRIEVAL_K          = int(os.getenv("RETRIEVAL_K", "2"))
OPENAI_KEY           = os.getenv("OPENAI_API_KEY")
MAX_HISTORY          = int(os.getenv("MAX_HISTORY", "10"))
TTL_SETUP            = int(os.getenv("TTL_SETUP", "1200"))
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT           = int(os.getenv("REDIS_PORT", "6379"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Validação de Variáveis Críticas ---
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX (ou PINECONE_INDEX_NAME) não configurada.")

# Redis
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

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

# módulo de memória
memory_store: Dict[str, List] = {}

# --- Prompt RAG ---
prompt_rag = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Você é o Assistente interno da Unimed para dúvidas de colaboradores.  
            Responda usando principalmente o conteúdo em "Contexto".  
            Se houver trechos relacionados, mesmo que parciais, utilize-os para formular uma resposta útil.  
            Evite inventar informações factuais (valores, percentuais, versões de documentos, missão, visão, benefícios) que não estejam no Contexto.

            ## Objetivo
            Fornecer respostas úteis, claras e confiáveis para apoiar o trabalho do dia a dia, com foco prático.

            ## Diretrizes
            - Utilize as evidências do Contexto da melhor forma possível, mesmo que incompletas.  
            - Só use fallback quando realmente não existir nenhum ponto relevante no Contexto.  
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

# ==================================================================
# Módulo de memória 
def get_memory(session_id):
    return memory_store.setdefault(session_id, [])

def clear_memory(session_id: str) -> bool:
    # return memory_store.pop(session_id, None) is not None
    return r.delete(f"session:{session_id}") > 0

# Módulo de memória usando Redis
def save_memory(session_id, messages, ttl=TTL_SETUP):
    r.setex(f"session:{session_id}", ttl, json.dumps(messages))

def load_memory(session_id):
    data = r.get(f"session:{session_id}")
    return json.loads(data) if data else []

def update_memory(session_id, messages):
    # truncated = messages[-MAX_HISTORY:]
    # memory_store[session_id] = truncated
    return messages[-MAX_HISTORY:]

# ==================================================================

def serialize_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    return [
        {
            "rank": i + 1,
            "text": (doc.page_content or "")[:],
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


def _fallback_variants(question: str, n: int) -> List[str]:
    """
    Fallback simples para gerar variações quando OpenAI não está disponível.
    """
    q = question.strip()
    base = [q]
    # Heurísticas básicas de reescrita
    extras = [
        q.lower(),
        q.upper(),
        # q.replace("qual ", "quais são ").replace(" o ", " "),
        # q.replace("como ", "qual é o procedimento para "),
        # q.replace("procedimento", "passo a passo"),
        # q.replace("valor", "custo"),
    ]
    # Dedup e limita a n
    out = []
    for s in base + extras:
        s = s.strip()
        if s and s not in out:
            out.append(s)
        if len(out) >= n:
            break
    return out

def generate_llm_variants(question: str, n: int = 4) -> List[str]:
    """
    Gera variações curtas da pergunta. Retorna lista de strings.
    Usa OpenAI se disponível; caso contrário, aplica um fallback local.
    """
    if not question or not question.strip():
        return []
    if client is None:
        return _fallback_variants(question, n)

    prompt = (
        f"Gere {n} variações curtas e diferentes da pergunta abaixo."
        "Use sinônimos e termos próximos, mas também expanda a pergunta com contextos prováveis "
        "(ex.: citar área/unidade responsável, documentos internos, públicos específicos, tipos de colaborador). "
        "As variações devem ser mais específicas que a pergunta original, sempre mantendo a intenção central. "
        "Inclua palavras como 'como', 'procedimento', 'área/unidade', 'ação/processo', 'quem', 'onde solicitar', 'quais critérios'. "
        "Se aplicável, substitua termos genéricos por sinônimos de glossário (ex.: treinamento = capacitação, curso, T&D). "
        "Se o termo for abreviação, substitua pelo nome por extenso."
        "Uma por linha, sem numeração.\n\n"
        f"Pergunta: {question}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "Você ajuda a reescrever perguntas de modo útil para busca."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response.choices[0].message.content or ""
        # Normaliza em lista e remove vazios/duplicados
        variants = [line.strip("•- \t") for line in content.splitlines() if line.strip()]
        # Garante que a pergunta original esteja presente
        if question.strip() not in variants:
            variants.insert(0, question.strip())
        
        # Dedup preservando ordem
        # variants = list(OrderedDict.fromkeys(variants))
        return variants[:n]
    except Exception as e:
        logger.warning("Erro ao gerar variações com OpenAI: %s", e)
        return _fallback_variants(question, n)



def retrieve_union(queries: List[str], k: int, namespace: str) -> List[Document]:
    """
    Executa recuperação para cada variação e faz a união deduplicada dos documentos.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "namespace": namespace},
    )
    collected: List[Document] = []
    seen = set()
    for q in queries:
        try:
            docs = retriever.get_relevant_documents(q)  # usa API explícita para string única
        except Exception as e:
            logger.error("Erro no retriever para '%s': %s", q, e)
            continue
        for d in docs or []:
            # Chave de deduplicação simples pelo conteúdo; ajuste se quiser usar metadados/IDs
            key = hash((d.page_content or "").strip())
            if key not in seen:
                seen.add(key)
                collected.append(d)
    return collected

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
    print("OK")
    return jsonify({"status": "ok"}), 200

@app.route("/chat", methods=["POST"])
def chat():
    print("Starting...")
    start_time = time.perf_counter()

    print("start time at:", start_time)
    try:
        payload = request.get_json(force=True, silent=False) or {}
        print("payload:", payload)
        question = (payload.get("question") or "").strip()
        print("Questão:", question)

        session_id = request.headers.get("X-Session-Id", "")
        
        if not session_id:
            return jsonify({"error":"Faltando session id"}), 400

        if not question:
            return jsonify({"error": "Campo 'question' é obrigatório."}), 400
        
        # memória
        memory = ConversationBufferMemory(memory_key="history", return_messages=True)
        # memory.chat_memory.messages = get_memory(session_id) # RAM
        memory.chat_memory.messages = load_memory(session_id) # Redis

        # Parâmetros opcionais
        k = int(payload.get("k") or RETRIEVAL_K)
        namespace = (payload.get("namespace") or DEFAULT_NAMESPACE or "default").strip()

        # logging.info(f"Pergunta: {question} | k={k} | ns={namespace} | sid={session_id}")

        # 1) Gera variações de consulta
        multi_query = generate_llm_variants(question, n=5)

        generate_question = random.choice(multi_query[1:])
        
        # 2) Recupera contexto (união deduplicada)
        context_docs: List[Document] = retrieve_union(multi_query, k=k, namespace=namespace)
        context_text = format_docs(context_docs)
        
        # 3) Gera resposta
        ai_msg: AIMessage = (prompt_rag | llm).invoke({
            "input"  : generate_question,
            "history": memory.chat_memory.messages,  # [],          # ajuste aqui se você passar histórico real
            "context": context_text
        })

        answer = _strip_fences_and_think(ai_msg.content)
        gen_info, usage_info = extract_gen_usage(ai_msg)

        # Memória
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        # memory_store[session_id] = memory.chat_memory.messages
        
        # salva no Redis com TTL e limite de histórico
        update_memory(session_id, memory.chat_memory.messages)
        save_memory(session_id, memory_store[session_id])

        # # Importante: jsonify não serializa AIMessage; retornamos o .content
        # answer_text = getattr(ai_msg, "content", "") if ai_msg else ""
        
        latency = (time.perf_counter() - start_time)*1000 # latencia em ms
        logger.info(f"Latência /chat = {latency:.2f} ms")
        print("FINISHED")
        return jsonify({
            "question_original": question,
            "question_used":generate_question,
            "answer": answer,
            "latency_ms":latency,
            "context": context_text,
            "retrieval": {
                "k": k,
                "namespace": namespace,
                "variants": multi_query,
                "docs": len(context_docs),
            },
            "metadata":{
                "generation_info":gen_info,
                "usage_info":usage_info
            }
        }), 200

    except Exception as e:
        logger.exception("Erro no endpoint /chat")
        return jsonify({"error": "Erro interno ao processar a solicitação.", "detail": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        session_id = payload.get("session_id")

        if not session_id:
            return jsonify({"error": "Campo 'session_id' é obrigatório."}), 400

        removed = clear_memory(session_id)

        if removed:
            return jsonify({"status": "ok", "message": f"Memória da sessão {session_id} foi limpa."}), 200
        else:
            return jsonify({"status": "not_found", "message": f"Nenhuma memória encontrada para {session_id}."}), 404

    except Exception as e:
        logger.exception("Erro no endpoint /clear")
        return jsonify({"error": "Erro interno ao limpar memória.", "detail": str(e)}), 500



if __name__ == "__main__":
    # Executar com: FLASK_ENV=development python main.py
    app.run(host="0.0.0.0", port=int(os.getenv("BACKEND_PORT", "8000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
