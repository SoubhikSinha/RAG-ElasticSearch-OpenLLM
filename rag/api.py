from fastapi import FastAPI
from pydantic import BaseModel
from rag.ingestion import download_pdfs_from_gdrive, process_pdfs
from rag.indexing import Indexer
from rag.retrieval import Retriever
from rag.generation import AnswerGenerator

# Init FastAPI
app = FastAPI(title="RAG System API")

# Initialize components
retriever = Retriever()
generator = AnswerGenerator(model_name="mistral")  # Ollama

# --------- MODELS ---------
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "hybrid"  # "elser" | "bm25" | "dense" | "hybrid"

class QueryResponse(BaseModel):
    answer: str
    citations: list

# --------- ENDPOINTS ---------

import requests

@app.get("/healthz")
def health_check():
    status = {"api": "ok"}

    # 1. Elasticsearch
    try:
        es_res = requests.get("http://localhost:9200")
        if es_res.status_code == 200:
            status["elasticsearch"] = "ok"
        else:
            status["elasticsearch"] = f"error ({es_res.status_code})"
    except Exception as e:
        status["elasticsearch"] = f"down ({str(e)})"

    # 2. Kibana
    try:
        kb_res = requests.get("http://localhost:5601")
        if kb_res.status_code == 200:
            status["kibana"] = "ok"
        else:
            status["kibana"] = f"error ({kb_res.status_code})"
    except Exception as e:
        status["kibana"] = f"down ({str(e)})"

    # 3. Ollama
    try:
        ollama_res = requests.get("http://localhost:11434/api/tags")
        if ollama_res.status_code == 200:
            status["ollama"] = "ok"
        else:
            status["ollama"] = f"error ({ollama_res.status_code})"
    except Exception as e:
        status["ollama"] = f"down ({str(e)})"

    return status


@app.post("/ingest")
def ingest():
    try:
        FOLDER_URL = "https://drive.google.com/drive/folders/1h6GptTW3DPCdhu7q5tY-83CXrpV8TmY_"
        local_dir = download_pdfs_from_gdrive(FOLDER_URL)
        if not local_dir:
            return {"error": "Failed to download PDFs from Google Drive"}

        # pass both arguments now
        docs = process_pdfs(local_dir, FOLDER_URL)
        if not docs:
            return {"error": "No documents were processed"}

        indexer = Indexer(index_name="rag_docs")
        indexer.create_index()
        indexer.index_documents(docs)

        return {"status": "ingestion complete", "docs_indexed": len(docs)}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}



@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question and get back answer + citations"""
    
    # Map retrieval mode
    if request.mode == "hybrid":
        retrieved = retriever.search_hybrid(request.question, top_k=request.top_k)
    elif request.mode == "elser":
        retrieved = retriever.search_elser(request.question, top_k=request.top_k)
    elif request.mode == "bm25":
        retrieved = retriever.search_bm25(request.question, top_k=request.top_k)
    elif request.mode == "dense":
        retrieved = retriever.search_dense(request.question, top_k=request.top_k)
    else:
        retrieved = []

    answer = generator.generate_answer(request.question, retrieved)

    # Return richer citations
    citations = [
        {
            "filename": doc["filename"],
            "url": doc["drive_url"],
            "snippet": doc["text"][:200]
        }
        for doc, _ in retrieved
    ]

    return {"answer": answer, "citations": citations}
