import os
import re
import hashlib
import unicodedata
from typing import List
from dotenv import load_dotenv

from itertools import product


from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeApiException



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
INDEX_NAME = os.getenv("PINECONE_INDEX", "teste")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # mxbai-embed-large = 1024



# ----------------------
# Helpers
# ----------------------
def slugify_ascii(text: str, keep: str = r"[^A-Za-z0-9._-]+") -> str:
    """Remove acentos e converte para ASCII seguro para IDs."""
    norm = unicodedata.normalize("NFKD", text)
    ascii_only = norm.encode("ascii", "ignore").decode("ascii")
    ascii_only = re.sub(keep, "_", ascii_only).strip("_")
    ascii_only = re.sub(r"_+", "_", ascii_only)
    return ascii_only or "doc"

def stable_doc_id(file_path: str) -> str:
    """Gera um doc_id estável para o PDF (slug + hash curto para evitar colisão)."""
    base = os.path.basename(file_path)
    slug = slugify_ascii(os.path.splitext(base)[0])
    h = hashlib.sha1(os.path.abspath(file_path).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"

def chunk_id(doc_id: str, i: int) -> str:
    # IDs ASCII, curtos e determinísticos; 512 chars é o limite duro do Pinecone
    return f"{doc_id}-c{str(i).zfill(5)}"[:128]

# ----------------------
# Store
# ----------------------
# class PineconeStore:
#     def __init__(self, sufixo):
#         api_key = os.getenv("PINECONE_API_KEY")
#         if not api_key:
#             raise ValueError("Variável de ambiente PINECONE_API_KEY não configurada")

#         self.pc = Pinecone(api_key=api_key)

#         # Cria índice se não existir
#         if INDEX_NAME not in self.pc.list_indexes().names():
#             print(f"Índice '{INDEX_NAME}_{sufixo}' não existe; criando…")
#             try:
#                 INDEX_NAME = INDEX_NAME+str(sufixo)
#                 self.pc.create_index(
#                     name=INDEX_NAME,
#                     dimension=EMBED_DIM,
#                     metric="cosine",
#                     spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
#                 )
#             except PineconeApiException as e:
#                 # 409 = already exists (race condition)
#                 if getattr(e, "status", None) != 409:
#                     raise
#         else:
#             print(f"Índice '{INDEX_NAME}' já existe.")

#         self.index = self.pc.Index(INDEX_NAME)
#         self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

class PineconeStore:
    def __init__(self, sufixo:str = None):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Variável de ambiente PINECONE_API_KEY não configurada")

        self.pc = Pinecone(api_key=api_key)
        
        if sufixo:
            # Define nome do índice com sufixo
            self.index_name = f"{INDEX_NAME}-{sufixo}"
        else:
            self.index_name = INDEX_NAME

        print(f"Index name: {self.index_name}")

        # Cria índice se não existir
        existing_indexes = self.pc.list_indexes()
        if self.index_name not in existing_indexes:
            print(f"Índice '{self.index_name}' não existe; criando…")
            try:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
            except PineconeApiException as e:
                # 409 = already exists (race condition)
                if getattr(e, "status", None) != 409:
                    raise
        else:
            print(f"Índice '{self.index_name}' já existe.")

        self.index = self.pc.Index(self.index_name)
        self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    
        
    # -------- Atualização segura (delete + upsert) --------
    def upsert_pdf(self, file_path: str, namespace: str = "default", delete_before: bool = True, batch_size: int = 100):
        # 1) Carregar e dividir
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", " \n \n", "***FIM***"])
        docs = splitter.split_documents(pages)

        if not docs:
            print(f"[WARN] Sem texto extraído de: {file_path}")
            return

        texts: List[str] = [d.page_content for d in docs]
        doc_id = stable_doc_id(file_path)

        # 2) (Opcional) Apagar vetores antigos desse PDF por filtro
        if delete_before:
            # metadado 'doc_id' é usado como alvo do filtro
            self.index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)

        # 3) Embeddings
        embeddings = self.embedder.embed_documents(texts)

        if len(embeddings[0]) != EMBED_DIM:
            raise ValueError(
                f"Dimensão do embedding ({len(embeddings[0])}) != EMBED_DIM ({EMBED_DIM}). "
                f"Confirme o modelo e o dimension do índice."
            )

        # 4) Preparar vetores (IDs ASCII seguros)
        vectors = []
        for i, emb in enumerate(embeddings):
            vid = chunk_id(doc_id, i)
            meta = {
                "doc_id": doc_id,                  # usado para update/delete por filtro
                "source": os.path.abspath(file_path),
                "page": docs[i].metadata.get("page"),
                "text": texts[i],
            }
            vectors.append({"id": vid, "values": emb, "metadata": meta})

        # 5) Upsert em lotes
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)

        print(f"[OK] {len(vectors)} chunks upsertados (namespace='{namespace}', doc_id='{doc_id}').")


    def delete_index(self):

        self.index.delete(ids=None, delete_all=True)

if __name__ == "__main__":



    chunk_size_values = [500, 1000, 1500, 2000, 3000]
    chunk_overlap_values = [200] # list(range(50, 300, 50))

    for CHUNK_SIZE, CHUNK_OVERLAP in product(chunk_size_values, chunk_overlap_values):
        sufixo = str(f"{CHUNK_SIZE}-{CHUNK_OVERLAP}")

        print(sufixo)

        store = PineconeStore(sufixo)

        

        

        # Upload um arquivo especifico
        store.upsert_pdf(
            file_path="downloads/DIR-324 - Diretriz de Beneficios - Rev.02.pdf",
            namespace="default",
            delete_before=False
        )


    # # Upload de todos os arquivos numa pasta
    # path = "downloads"
    # for fname in os.listdir(path):
    #     fpath = os.path.join(path, fname)
    #     if not os.path.isfile(fpath):
    #         continue
    #     # pule não-PDFs
    #     if not fname.lower().endswith(".pdf"):
    #         continue
    #     try:
    #         store.upsert_pdf(fpath, namespace="default", delete_before=False)
    #     except Exception as e:
    #         print(f"[ERRO] {fname}: {e}")
