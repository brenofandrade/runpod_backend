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



# --- query_pipeline.py ---
import re
import unicodedata
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor


# LangChain:
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoderReranker
from langchain_core.documents import Document


# --- Variáveis de Ambiente ---
OLLAMA_BASE_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL  = os.getenv("GENERATION_MODEL", "llama3.2:latest")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large:latest")
# CORREÇÃO: pegue PINECONE_API_KEY corretamente, sem defaults enganosos
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY_DSUNIBLU")
# PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY_DSUNIBLU")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")



# Seu retriever vetorial (Pinecone)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


# instância 


llm = ChatOllama()
embedding = OllamaEmbeddings()






# I. Indexing


# 1) Load


# 2) Split


# 3) Embed


# 4) Store




# II. Retrieving and Generating

# 1) Question
#   a) Normalizar a pergunta
#   b) Multi-Query



# 2) Retrieve
#   a) Busca por termo
#   b) Busca semântica
#   c) Busca híbrida
#   d) Threshold dinâmico: Se top-1 cai abaixo de min_score, então k aumenta e/ou gerar novas variações


# 3) Prompt
#   a) Normalizar PT-BR: remover acentos, singular, plural simples, "vale-refeição/VR"
#   b) Dicionário de sinônimos e siglas internas
#       Clube Ipiranga, Parceria clube, benefício clube
#       Vale-refeição, VR
#       Vale-alimentação, VA
#       Plano odontológico, odonto, uniodonto


# 4) LLM


# 5) Answer



# === 1) Normalização básica PT-BR ===
_PT_STOPWORDS = {"de","da","do","das","dos","para","por","com","sem","a","o","as","os","em","no","na","nos","nas","e","ou","um","uma","uns","umas"}



def normalize_pt(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text

def canonicalize_numbers(text: str) -> str:
    # Troca "mil e trinta e dois" -> (opcional) ou normaliza R$ 1.032,00 -> 1032.00 etc.
    text = text.replace("R$ ", "R$").replace(",", ".")
    return text

def expand_siglas(text: str) -> List[str]:
    # Regras de casa: expanda VR, VA, odonto, etc.
    pairs = {
        "vr": "vale refeicao",
        "va": "vale alimentacao",
        "odonto": "plano odontologico",
        "gnc": "cinema gnc",
    }
    variants = {text}
    norm = normalize_pt(text)
    for k, v in pairs.items():
        if re.search(rf"\b{k}\b", norm):
            variants.add(norm.replace(k, v))
    return list(variants)


# === 2) Multi-Query + HyDE ===
def generate_llm_variants(llm, question: str, n: int = 4) -> List[str]:
    prompt = f"""
Gere {n} variações curtas e diferentes da pergunta abaixo, cobrindo sinônimos, termos próximos,
e formas que um colaborador poderia usar. Uma por linha, sem numeração.

Pergunta: {question}
"""
    out = llm.invoke(prompt).strip().splitlines()
    out = [x.strip("-• ").strip() for x in out if x.strip()]
    return out[:n]