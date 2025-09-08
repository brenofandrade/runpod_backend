# main.py
# Para servir a aplicação com waitress:
# waitress-serve --listen=0.0.0.0:8000 main:app

import os
import json
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
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pydantic import BaseModel, Field

# ---- Config base ----
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

# Falha cedo se variáveis críticas faltarem
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY não configurada.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("PINECONE_INDEX_NAME não configurada.")

# ROUTER_SYSTEM = """Você é um roteador de perguntas para um sistema RAG.
# Sua tarefa é decidir se a pergunta do usuário precisa consultar a base de conhecimento (RAG)
# ou se pode ser respondida apenas com conhecimento geral do modelo.

# Regras:
# - Marque use_rag=True quando a pergunta:
#   * pede fatos específicos da organização, documentos internos, bases privadas, PDFs, planilhas, contratos, manuais, políticas, repositórios de código, logs;
#   * pede "fontes", "documentos", "onde está", "segundo o manual X", "na pasta Y", "no contrato Z";
#   * depende de detalhes locais: nomes/IDs internos, datas específicas, versões internas, números de chamado, etc.
# - Marque use_rag=False quando a pergunta:
#   * é conversa/etiqueta (saudações, agradecimentos, etc.),
#   * é conhecimento geral comum (explicações conceituais, matemática simples, Python/SQL genérico), 
#   * é opinião/brainstorm sem citar material interno.

# Se houver ambiguidade, prefira use_rag=True.
# Responda **apenas** JSON válido com os campos do schema.
# """

ROUTER_SYSTEM = """Você é um roteador de perguntas para um sistema RAG.
Decida se a pergunta precisa consultar a base (RAG) ou se pode ser respondida sem RAG.

Regras:
- use_rag=True quando a pergunta depende de documentos internos, PDFs, planilhas, contratos, manuais, políticas, repositórios, logs, pede "fontes"/"documentos"/"onde está", ou cita artefatos internos (IDs, versões, datas, chamados).
- use_rag=False para saudações/etiqueta, conhecimento geral, explicações comuns, brainstorm genérico.

Se houver ambiguidade, prefira use_rag=True.

Responda **apenas** com um objeto JSON válido, sem markdown, sem comentários, exatamente com estes campos:
- "use_rag": booleano (true/false)
- "confidence": número entre 0 e 1
- "rationale": string curta em PT-BR

Exemplo de formato (exemplo, não copie valores):
{"use_rag": true, "confidence": 0.82, "rationale": "Cita um manual interno"}
"""

class RouteDecision(BaseModel):
    use_rag:bool = Field(..., description="Se deve acionar r RAG.")
    confidence:float = Field(..., ge=0, le=1, description="Confiança da decisão.")
    rationale:str = Field(..., description="Breve justificativa em PT-Br")

# router_llm = ChatOllama(model="llama3.2", temperature=0)
router_llm = ChatOllama(model="llama3.2", temperature=0, base_url=OLLAMA_BASE_URL)
direct_llm = ChatOllama(model=GENERATION_MODEL, temperature=0, base_url=OLLAMA_BASE_URL)
router_parser = JsonOutputParser(pydantic_object=RouteDecision)
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_SYSTEM),
        ("user", "Pergunta: Onde está o manual de onboarding da equipe X?"),
        ("assistant", '{"use_rag": true, "confidence": 0.86, "rationale": "Pede documento interno"}'),
        ("user", "Pergunta: Explique o que é normalização em bancos de dados."),
        ("assistant", '{"use_rag": false, "confidence": 0.77, "rationale": "Conhecimento geral"}'),
        ("user", "Pergunta: {question}\nDevolva **apenas** o objeto JSON preenchido, nada além.")
    ]
)

direct_system = """Você é um assistente técnico. 
Responda de forma objetiva e correta. 
Se não tiver certeza, diga que não sabe. Em PT-BR."""
direct_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", direct_system),
        ("user", "{question}")
    ]
)
direct_llm = ChatOllama(model=GENERATION_MODEL, temperature=0)  
direct_chain = direct_prompt | direct_llm

def _looks_like_json_schema(d: dict) -> bool:
    return isinstance(d, dict) and (
        ("properties" in d and "type" in d) or
        (set(d.keys()) <= {"type", "properties", "required", "title", "$schema"})
    )

# def llm_route(question: str) -> RouteDecision:
#     msg = router_prompt.invoke({"question": question})
#     out = router_llm.invoke(msg)
#     return router_parser.parse(out.content)

# def llm_route(question: str) -> RouteDecision:
#     msg = router_prompt.invoke({"question": question})
#     out = router_llm.invoke(msg)

#     logging.debug(f"[Router LLM] Raw output type: {type(out)} | Output: {out}")
    
#     raw_content = getattr(out, "content", out)  # Se 'out' for string direto
#     logging.debug(f"[Router LLM] Raw content: {raw_content} | Type: {type(raw_content)}")

#     if not isinstance(raw_content, str):
#         raise ValueError("Resposta do router não é uma string JSON válida.")
    
#     try:
#         parsed = router_parser.parse(raw_content)
#     except Exception as e:
#         logging.exception(f"[Router Parser] Falha ao fazer parse da resposta: {e}")
#         raise

#     logging.debug(f"[Router Parser] Parsed type: {type(parsed)} | Parsed value: {parsed}")

#     if not isinstance(parsed, RouteDecision):
#         raise TypeError(f"Resposta do router não é um objeto RouteDecision válido. Recebido: {type(parsed)}")

#     return parsed


###### Parser manual
# def llm_route(question: str) -> RouteDecision:
#     msg = router_prompt.invoke({"question": question})
#     out = router_llm.invoke(msg)

#     raw = getattr(out, "content", out)
#     if raw is None:
#         raise ValueError("Router LLM retornou vazio (None). Verifique o Ollama/base_url/modelo.")
    
#     if not isinstance(raw, str):
#         # alguns providers retornam objetos Message; tenta converter
#         raw = str(raw)
#     logging.debug(f"[Router LLM] bruto: {raw!r}")

#     cleaned = _strip_fences_and_think(raw)
#     candidate = _extract_first_json_object(cleaned) or cleaned
#     if not candidate:
#         raise ValueError("Router LLM não retornou JSON reconhecível.")
    

#     try:
#         parsed = router_parser.parse(candidate)
#         if isinstance(parsed, RouteDecision):
#             return parsed
#     except Exception as e:
#         logging.warning(f"[Router Parser] Falhou, tentando json.loads. Erro: {e}")

#     # 2ª tentativa: json.loads + pydantic
#     try:
#         parsed_json = json.loads(candidate)
#         return RouteDecision(**parsed_json)
#     except Exception as e:
#         logging.error(f"[Router] Conteúdo impossível de parsear em JSON: {candidate!r}")
#         raise

def llm_route(question: str) -> RouteDecision:
    msg = router_prompt.invoke({"question": question})
    out = router_llm.invoke(msg)

    raw = getattr(out, "content", out)
    if raw is None:
        raise ValueError("Router LLM retornou vazio (None). Verifique o Ollama/base_url/modelo.")

    if not isinstance(raw, str):
        raw = str(raw)

    logging.debug(f"[Router LLM] bruto: {raw!r}")

    cleaned = _strip_fences_and_think(raw)
    candidate = _extract_first_json_object(cleaned) or cleaned
    if not candidate:
        raise ValueError("Router LLM não retornou JSON reconhecível.")

    # 1ª tentativa: parser do LangChain
    try:
        parsed = router_parser.parse(candidate)
        if isinstance(parsed, RouteDecision):
            return parsed
    except Exception as e:
        logging.warning(f"[Router Parser] Falhou, tentando json.loads. Erro: {e}")

    # 2ª tentativa: JSON bruto
    parsed_json = json.loads(candidate)

    # Se o modelo devolveu um SCHEMA (properties/required/type), falhe com mensagem clara
    if _looks_like_json_schema(parsed_json):
        logging.error(f"[Router] O modelo retornou um SCHEMA em vez de um OBJETO: {parsed_json}")
        raise ValueError(
            "O roteador devolveu um JSON Schema em vez de um objeto com campos {use_rag, confidence, rationale}. "
            "Ajuste o prompt para exigir o objeto preenchido (vide ROUTER_SYSTEM) ou acrescente um exemplo."
        )

    # Caso seja um objeto, prosseguir
    return RouteDecision(**parsed_json)


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

import re

def _strip_fences_and_think(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # remove <think>...</think>
    import re
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL)
    # remove cercas ```json ... ``` ou ``` ... ```
    s = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()

def _extract_first_json_object(s: str) -> str:
    # pega o primeiro {...} balanceado básico
    import re
    # tentativa simples: do primeiro { ao último } se só houver um objeto
    first = s.find("{")
    last  = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first:last+1].strip()
    # fallback: regex gulosa mínima
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0).strip() if m else ""



# ---- App ----
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
# CORS(app)
app.url_map.strict_slashes = False

# ---- Recursos externos (Pinecone / Ollama / LangChain) ----
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

llm = ChatOllama(model=GENERATION_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

# Instanciamos o VectorStore sem travar namespace; passaremos namespace no retriever.
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

SYSTEM_PROMPT = ("""Você é um assistente especializado em Responder Apenas com Base no Contexto.

Regras:
1) Use exclusivamente o conteúdo em “Contexto”.
2) Seja conciso (<=120 palavras, exceto código/tabelas).
3) Se não souber, diga: "Não sei."
4) Cite fontes no formato [fonte:source|doc_id|p.page] quando disponíveis.
5) Se houver conflito no contexto, informe o conflito sem extrapolar.
6) Responda em português. Não revele o raciocínio.

# Contexto:
{context}

# Pergunta:
{input}

# Resposta:
"""                 
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
        print("[OK] - Chamada recebida")
        # Verificação leve de Ollama (chamada mínima via geração “ping”)
        # _ = llm.invoke("pong?")  # sem contexto, só sanity check
        return jsonify({"status": "OK"}), 200
    except Exception as e:
        logging.exception("Health check falhou: %s", e)
        return jsonify({"status": "DEGRADED", "error": str(e)}), 503




@app.route("/chat", methods=["POST", "OPTIONS"])
@app.route("/chat/", methods=["POST", "OPTIONS"])
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    question: str = (payload.get("question") or "").strip()
    session_id = request.headers.get("X-Session-Id")

    if not session_id:
        return jsonify({"error":"Faltando session_id"}), 400
    print(f"Session ID: {session_id}")

    if not question:
        return jsonify({"error": "Campo 'question' é obrigatório."}), 400
    


    # parâmetros opcionais
    k = int(payload.get("k") or 4)
    namespace = (payload.get("namespace") or DEFAULT_NAMESPACE).strip()
    
    force_rag = bool(payload.get("force_rag") or False)
    force_direct = bool(payload.get("force_direct") or False)
    
    if force_rag and force_direct:
        return jsonify({"error": "Não use force_rag e force_direct ao mesmo tempo."}), 400


    logging.info(f"Question: {question} | k={k} | namespace={namespace!r} | force_rag={force_rag} | force_direct={force_direct}")

    # logging.info(f"Question: {question} | k={k} | namespace={namespace!r}")


    try:
        if not force_rag and not force_direct:
            decision = llm_route(question)
            # logging.info(f"[Router] use_rag={decision.use_rag} conf={decision.confidence:.2f} rationale={decision.rationale}")
            use_rag = decision.use_rag
        else:
            use_rag = force_rag and not force_direct

        if not use_rag:
            # Responda diretamente (sem RAG)
            mensagem = direct_chain.invoke({"question":question})

            resposta_direta = getattr(mensagem, "content", "").strip()
            resposta_final = _strip_fences_and_think(resposta_direta)
            return jsonify({
                "answer":resposta_final,
                "sources":[],
                "namespace":None,
                "k":0,
                "user_rag": False,
            }), 200
        
        # search_kwargs = {"k": k, "namespace": namespace}
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",  # troque para "similarity_score_threshold" se disponível
        #     search_kwargs=search_kwargs,
        # )

        # rag_chain = create_retrieval_chain(retriever, combine_chain)
        # result = rag_chain.invoke({"input": question})

        # answer = (result.get("answer") or "").strip()
        # final_answer = extract_final_answer(answer)

        # context_docs = result.get("context", []) or []

        # # Fallback inteligente: se o contexto vier vazio, tenta resposta direta
        # if not context_docs:
        #     logging.info("[RAG] Nenhum documento relevante. Fallback para resposta direta.")
        #     direct_msg = direct_chain.invoke({"question": question})
        #     direct_answer = getattr(direct_msg, "content", "").strip()
        #     final_answer = extract_final_answer(direct_answer) if "extract_final_answer" in globals() else direct_answer

        # response = {
        #     "answer": final_answer,
        #     "sources": serialize_sources(context_docs),
        #     "namespace": namespace,
        #     "k": k,
        #     "used_rag": bool(context_docs),  # sinaliza se realmente houve contexto
        # }
        # return jsonify(response), 200

        
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "namespace": namespace},
        )

        rag_chain = create_retrieval_chain(retriever, combine_chain)

        # Chave correta para o chain: "input"
        result = rag_chain.invoke({"input": question})
        answer = result.get("answer", "").strip()
        final_answer = _strip_fences_and_think(answer)
        context_docs = result.get("context", [])

        if not context_docs:
            logging.info("[RAG] Sem documentos relevantes. Fazendo fallback para resposta direta.")
            direct_msg = direct_chain.invoke({"question": question})
            final_answer = _strip_fences_and_think(getattr(direct_msg, "content", "") or "Não sei.")

        response = {
            "answer": final_answer,
            "sources": serialize_sources(context_docs),
            "namespace": namespace,
            "k": k,
            "used_rag": bool(context_docs),
        }
        return jsonify(response), 200

    except Exception as error:
        logging.exception("Erro no chat: %s", error)
        return jsonify({"error": str(error)}), 500

    # try:
    #     # retriever por request (pra aceitar k/namespace)
    #     retriever = vectorstore.as_retriever(
    #         search_type="similarity",
    #         search_kwargs={"k": k, "namespace": namespace},
    #     )

    #     rag_chain = create_retrieval_chain(retriever, combine_chain)

    #     # Chave correta para o chain: "input"
    #     result = rag_chain.invoke({"input": question})

    #     answer = result.get("answer", "").strip()

    #     final_answer = extract_final_answer(answer)

    #     context_docs = result.get("context", [])

    #     response = {
    #         "answer": final_answer,
    #         "sources": serialize_sources(context_docs),
    #         "namespace": namespace,
    #         "k": k,
    #     }
    #     return jsonify(response), 200

    # except Exception as e:
    #     logging.exception("Erro no chat: %s", e)
    #     return jsonify({"error": str(e)}), 500






# ---- Execução direta (dev) ----
if __name__ == "__main__":
    # Em produção prefira waitress-serve (ver cabeçalho).
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=False)
