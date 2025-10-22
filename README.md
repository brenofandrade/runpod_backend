# runpod_backend# Flask RAG API (LangChain + Ollama + Pinecone)
## Variáveis de ambiente


Crie um arquivo `.env` na raiz com:


```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
GENERATION_MODEL=llama3.2
EMBEDDING_MODEL=mxbai-embed-large


# Pinecone
PINECONE_API_KEY=seu_api_key
PINECONE_INDEX_NAME=pdf-vector-store
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=default


# App
LOG_LEVEL=INFO
PORT=8080
```


> A dimensão do índice é detectada automaticamente via `Ollama /api/show`. Se não for possível, usa `1024` (compatível com `mxbai-embed-large`).


## Instalação


```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


## Rodando


```bash
export FLASK_APP=app.py
flask run -p 8080
# ou
python app.py
```


## Testes rápidos


Saúde:
```bash
curl -s http://localhost:8080/health | jq
```


Pergunta simples:
```bash
curl -s -X POST http://localhost:8080/chat \
-H 'Content-Type: application/json' \
-d '{
"question": "Quais políticas de férias?",
"k": 4
}' | jq
```


Pergunta + ingestão ad hoc:
```bash
curl -s -X POST http://localhost:8080/chat \
-H 'Content-Type: application/json' \
-d '{
"question": "Qual é o SLA?",
"namespace": "demo",
"documents": [
{"text": "SLA de atendimento crítico: resposta em 1h, solução em 8h", "metadata": {"source":"sla.md"}}
]
}' | jq
```


## Notas


- O `POST /chat` retorna `answer`, `sources` (trechos + metadados) e `meta`.
- Para PDFs, crie um pipeline separado de ingestão (loader + splitter) e alimente o `vectorstore.add_documents` com `Document`s.
- Caso use Docker, exponha a porta 11434 do Ollama e certifique-se de que a rede permite o acesso do contêiner ao host.


--- 

# Manual de Utilização do assistente de conversação para documentos internos

## Visão geral




## Para que serve este assistente?



## Público-alvo

[X] Colaboradores
[ ] RH
[ ] Atendimento ao cliente
[ ] Outras áreas: ________________

---

## Como funciona?

### Conceitos-chave





---


## FAQ - Perguntas Frequentes


1. O que posso perguntar ao assistente?
Você pode fazer perguntas sobre políticas internas, benefícios, processos operacionais, normas e procedimentos da empresa.

2. O assistente responde todas as questões?
Não. Ele responde apenas com base nos documentos existentes em sua base de conhecimento. Se não houver contexto suficiente, uma mensagem de contingência será exibida.

3. Como sei se a resposta está correta?
As fontes onde o assitente procurou a resposta para sua pergunta serão exibidas logo abaixo. Você poderá verificar os trechos originais.

4. Posso confiar 100% nas respostas?
AS respostas são baseadas em documentos oficiais, mas sempre valide com os documentos originais e em caso de dúvida, com as áreas de negócio.

5. Posso sugerir melhorias?
Sim. Use a sessão de feedback ao final da página para enviar uma mensagem.


## Atualizações e versões

| Data       | Versão | Alteração                            |
|------------|--------|--------------------------------------|
| 2025-10-06 | 0.0.1  | Testes de usabilidade e documentação |

---


## Fluxo

Usuário
  │
  ▼
Interface (Chat/UI)
  │  pergunta
  ▼
Servidor/API ── pré-processa ──► Retriever (Vector Store)
  │                               │
  │                               └──► top-k trechos (contexto)
  │
  └── monta prompt RAG (instruções + pergunta + contexto)
           │
           └──► LLM (gera resposta) 
                     │                           
             ┌───────┴────────┐                  
             │ contexto ok?   │                 
             └───────┬────────┘
                     │Sim
                     ▼
             Resposta + fontes
                     │
                     ▼
            Interface (exibe + feedback)
                     │
                     ▼
            Logs & Métricas (latência, p95, tokens, fallback, feedback)

