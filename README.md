# RAG (ElasticSearch + OpenLLM)

### _End-to-end Retrieval-Augmented Generation powered by Elasticsearch, ELSER, BM25 and Dense Embeddings_

<p align="center"> <img src="https://static-www.elastic.co/v3/assets/bltefdd0b53724fa2ce/blt2cb6cab4deba98f9/6671d4b15cb7a3ca2fbfd9fa/illustrations-rag-workflows-with-elastic-elasticsearch-logo-brain.png" width="600" alt="Elastic RAG System Architecture"> </p>

<p align="center"> <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a> <a href="https://www.elastic.co/elasticsearch/"><img src="https://img.shields.io/badge/Elasticsearch-9.1.2-005571?logo=elasticsearch" alt="ElasticSearch"></a> <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi" alt="FastAPI"></a> <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-UI-E84C3D?logo=streamlit" alt="Streamlit"></p>

<br>

This is a **production-ready Retrieval-Augmented Generation pipeline** built on top of the **Elastic Stack**.  
It brings together:
-   **Hybrid Search** → BM25 + ELSER (sparse embeddings) + Dense Embeddings (MiniLM) with Reciprocal Rank Fusion
-   **Multi-modal Ingestion** → PDF ingestion from Google Drive, text extraction, token-level chunking
-   **LLM-based Answering** → Powered by local Ollama model (Mistral) with grounding + guardrails
-   **Developer-friendly APIs & UI** → REST endpoints (FastAPI) + lightweight Streamlit interface

In short: _a plug-and-play RAG system that shows how to combine Elastic’s search power with modern embeddings & LLMs_.

<br>

## ✨ Why this project?
-   🔍 **Search relevance**: Elastic’s **ELSER** bridges BM25 and dense retrieval for higher recall.
-   🧠 **Grounded generation**: Answers are built strictly from retrieved evidence — no hallucinations.
-   ⚡ **Scalable infra**: Elasticsearch 9.1.2 + Kibana + ML nodes, all containerized with Docker.
-   🎯 **End-to-end example**: From ingestion → indexing → retrieval → generation → UI.

<br>

## 📑 Table of Contents
- [📸 Demo](#-demo)
- [📖 Overview](#-overview)
- [⚡ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠 Tech Stack](#-tech-stack)
- [⚙️ Setup Instructions](#️-setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Environment Setup](#environment-setup)
  - [Start Elasticsearch + Kibana](#start-elasticsearch--kibana)
  - [Run Ingestion](#run-ingestion)
  - [Start API](#start-api)
  - [Launch UI](#launch-ui)
- [📂 Project Structure](#-project-structure)
- [🙏 Acknowledgments](#-acknowledgments)

<br>

## 📸 Demo
<img src="https://github.com/SoubhikSinha/RAG-ElasticSearch-OpenLLM/blob/main/DemoPics/Screenshot%202025-08-28%20at%2018.16.46.png" width="100%" />
<img src="https://github.com/SoubhikSinha/RAG-ElasticSearch-OpenLLM/blob/main/DemoPics/Screenshot%202025-08-28%20at%2018.20.11.png" width="100%" />
<img src="https://github.com/SoubhikSinha/RAG-ElasticSearch-OpenLLM/blob/main/DemoPics/Screenshot%202025-08-28%20at%2018.25.39.png" width="100%" />

<br>

## 📖 Overview
Modern LLMs are powerful but inherently limited: they hallucinate, forget domain-specific knowledge, and cannot reason over large external datasets by themselves. **Retrieval-Augmented Generation (RAG)** solves this by combining search with generation — fetching relevant documents and grounding model outputs in real evidence.

This project demonstrates an **end-to-end, production-ready RAG system** built on **Elasticsearch 9.1.2**, leveraging its search primitives alongside open-source LLM tooling. It goes beyond toy examples by integrating three complementary retrieval strategies:
-   **BM25** → keyword-based search for exact lexical matches.
-   **ELSER (Elastic Learned Sparse Encoder)** → ML-powered sparse vectors for semantic relevance.
-   **Dense embeddings (MiniLM)** → neural embeddings for fine-grained semantic similarity.

These signals are fused together with **Reciprocal Rank Fusion (RRF)** to maximize both precision and recall.

Once documents are retrieved, a **local or open LLM (via HuggingFace or Ollama)** synthesizes the final answer, constrained to the retrieved evidence. If no strong context exists, the system explicitly replies with _“I don’t know.”_ This ensures reliability and prevents hallucinations.

The project includes:
-   **Ingestion pipeline** for PDFs from Google Drive, with text extraction, token-level chunking, and metadata enrichment.
-   **Indexing pipeline** that encodes chunks with both sparse and dense models.
-   **Retrieval layer** supporting ELSER-only, dense-only, or hybrid fusion.
-   **Guardrails** for off-topic or unsafe queries.
-   **FastAPI backend** exposing `/query`, `/ingest`, `/healthz` endpoints.
-   **Streamlit UI** for interactive exploration with answers + citations.

In short: this repository shows how to build a **scalable, explainable, and developer-friendly RAG pipeline** using Elasticsearch as the backbone.

<br>

## ⚡ Features
### 📂 Ingestion
-   **Google Drive Integration** – Seamlessly load PDFs from a shared Drive folder.
-   **Text Extraction** – Parse PDF content using `PyPDF2` (with OCR-ready hooks for scanned files).
-   **Smart Chunking** – Split text into ~300-token segments with overlap for context retention.
-   **Rich Metadata** – Each chunk stores filename, Drive URL, and chunk ID for traceability.
    
----------

### 🧠 Indexing
-   **BM25 Baseline** – Store raw text in a `text` field for keyword search.
-   **ELSER Encoding** – Expand text into sparse semantic features (`text_expansion`) using Elastic’s ML model.
-   **Dense Embeddings** – Encode chunks with `sentence-transformers/all-MiniLM-L6-v2` for neural similarity.
-   **Unified Index** – All signals live in a single index with explicit mappings.

----------


### 🔍 Retrieval
-   **BM25-only Mode** – Classic keyword-based retrieval for exact lexical matches.
-   **ELSER-only Mode** – Semantic sparse retrieval using Elastic’s ML-powered encoder.
- **Dense-only Mode** – Neural retrieval using `sentence-transformers/all-MiniLM-L6-v2` embeddings with cosine similarity.
-   **Hybrid Mode** – Reciprocal Rank Fusion (RRF) combining BM25, ELSER, and dense embeddings for maximum recall and precision.
-   **Configurable Top-k** – Adjustable candidate size (`k`, default = 5).
    
----------

### 💬 Answer Generation
-   **Local/Open LLMs** – Integrate with Ollama backend.
-   **Grounded Prompts** – Answers are constructed only from retrieved context.
-   **Hallucination Control** – If no strong evidence is found, respond with _“I don’t know.”_
-   **Guardrails** – Reject unsafe, harmful, or off-topic queries.
    
----------

### ⚡ API
-   **FastAPI Endpoints** –
    -   `POST /query` → submit a question, get answer + citations.
    -   `POST /ingest` → re-index documents from Google Drive.
    -   `GET /healthz` → health check.
-   **JSON-first Design** – Easy integration with downstream apps.
    
----------

### 🎨 UI

-   **Streamlit Frontend** – Clean web interface for interactive exploration.
-   **Question Box** – Type any question, see instant answers.
-   **Citations** – Display title, snippet, and Drive link for each supporting doc.
-   **Retrieval Toggle** – Switch between ELSER-only and Hybrid retrieval modes.
    
----------

### 🔒 Reliability & Scalability
-   **Dockerized Stack** – Elasticsearch 9.1.2 + Kibana + FastAPI + Streamlit, all container-ready.
-   **ML-enabled Nodes** – Runs ELSER seamlessly inside Elastic’s ML runtime.
-   **Extensible Design** – Plug in new embedding models, ingestion sources, or UIs without re-architecture.

<br>

## 🏗️ Architecture
The system is designed as a modular pipeline, where each stage is independent but seamlessly connected.

1.  **Ingestion**
    -   PDFs are fetched from a shared **Google Drive folder**.
    -   Text is extracted, split into ~300-token overlapping chunks.
    -   Each chunk is enriched with metadata: filename, Drive URL, chunk ID.
2.  **Indexing**
    -   **BM25**: raw text stored in a `text` field for classic keyword search.
    -   **ELSER**: chunks expanded into sparse semantic features (`text_expansion`) using Elastic’s ML model.
    -   **Dense Embeddings**: vectors generated with `all-MiniLM-L6-v2` for semantic similarity.
    -   Unified index in **Elasticsearch 9.1.2** stores all signals.  
3.  **Retrieval**
    -   **BM25-only**: keyword search.
    -   **ELSER-only**: semantic sparse retrieval.  
    -   **Dense-only**: embedding similarity search.
    -   **Hybrid**: Reciprocal Rank Fusion (RRF) combining BM25 + ELSER + Dense for maximum recall.  
4.  **Answer Generation**
    -   Top-k results are merged into a context window.
    -   A local/open **LLM (HuggingFace / Ollama)** generates an answer grounded in evidence.
    -   If context is weak → system responds with _“I don’t know.”_
    -   Guardrails enforce safe, relevant outputs. 
5.  **Serving Layer**
    -   **FastAPI** backend exposes REST endpoints:
        -   `POST /ingest` → (re)load & index Drive docs
        -   `POST /query` → answer a question with citations 
        -   `GET /healthz` → health check
    -   **Streamlit UI**: interactive front-end for querying, answer display, citation visualization, and retrieval-mode toggling.

<br>

### High-Level Flow (ASCII Diagram)
                ┌───────────────────────────┐
                │     Google Drive PDFs     │
                └─────────────┬─────────────┘
                              │
                     Ingestion & Chunking
                              │
                ┌─────────────▼─────────────┐
                │   Elasticsearch Index     │
                │ ───────────────────────── │
                │  • BM25 (text field)      │
                │  • ELSER (sparse vectors) │
                │  • Dense vectors (MiniLM) │
                └─────────────┬─────────────┘
                              │
                     Retrieval Strategies
       ┌───────────────┬───────────────┬───────────────┐
       │               │               │               │
    BM25-only      ELSER-only      Dense-only       Hybrid (RRF)
       └───────────────┴───────────────┴───────────────┘
                              │
                        Top-k Results
                              │
                ┌─────────────▼─────────────┐
                │     Answer Generator      │
                │ (HuggingFace / Ollama LLM)│
                └─────────────┬─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
    FastAPI API           Streamlit UI          Kibana Monitoring


<br>

## 🛠 Tech Stack
### 🔹 Core Infrastructure
-   **[Elasticsearch 9.1.2](https://www.elastic.co/elasticsearch/?utm_source=chatgpt.com)** → Search backbone, powering BM25, ELSER (sparse semantic search), and dense vector retrieval.
-   **[Kibana 9.1.2](https://www.elastic.co/kibana/?utm_source=chatgpt.com)** → Monitoring, querying, and visualizing ingestion and retrieval pipelines.
-   **Docker / Docker Compose** → Containerized deployment of Elasticsearch, Kibana, API, and UI services.
    
----------

### 🔹 Machine Learning & Retrieval
-   **ELSER** → Elastic’s Learned Sparse Encoder for semantic sparse retrieval (`text_expansion`).
-   **sentence-transformers/all-MiniLM-L6-v2** → Dense embeddings (384-dimensional vectors) for semantic similarity search.
-   **Reciprocal Rank Fusion (RRF)** → Hybrid ranking strategy combining BM25, ELSER, and dense vectors.
    
----------

### 🔹 Answer Generation
-   **Ollama** → Local LLM runtime for running open models efficiently on Mac.
-   **Mistral** → Lightweight, high-performance open LLM used via Ollama for grounded answer generation.
-   Guardrails ensure answers are safe, relevant, and fallback to _“I don’t know”_ if evidence is weak.
    
----------

### 🔹 Backend & API
-   **FastAPI** → REST API with endpoints for querying (`/query`), ingestion (`/ingest`), and health checks (`/healthz`).
-   **Uvicorn** → ASGI server for FastAPI.

----------

### 🔹 Frontend & UI
-   **Streamlit** → Lightweight web interface for user queries, answers, and citations.
    
----------

### 🔹 Data Processing
-   **[PyPDF2](https://pypi.org/project/pypdf2/?utm_source=chatgpt.com)** → Extract text from PDFs.
-   **[gdown](https://github.com/wkentaro/gdown?utm_source=chatgpt.com)** → Download files and folders from Google Drive.
-   **[python-dotenv](https://pypi.org/project/python-dotenv/?utm_source=chatgpt.com)** → Manage environment variables securely (`.env`).
    
----------

### 🔹 Language & Runtime
-   **Python 3.10+** → Core language for ingestion, indexing, retrieval, and orchestration.

<br>

## ⚙️ Setup Instructions
### Prerequisites
This project was built and tested on a **MacBook M3 (Apple Silicon, ARM64)**.  
It should run on other systems (Linux, Windows) with minor adjustments, but Apple Silicon users should pay special attention to the `--platform=linux/amd64` flag when running Elasticsearch/Kibana, since **ML features (ELSER)** are not fully supported in the ARM builds.
<br>

Before you begin, make sure you have:
-   **[Docker Desktop](https://www.docker.com/)** (latest version)
    -   Required to run **Elasticsearch 9.1.2** and **Kibana 9.1.2** containers.
    -   Allocate at least **6–8 GB of RAM** to Docker for ML models (ELSER) to load properly.
-   **[Anaconda](https://www.anaconda.com) / Miniconda**
    -   Recommended for creating an isolated Python environment.
-   **Python 3.10+** (managed via Anaconda or pyenv)
    -   Required to run ingestion, indexing, and API/UI code.
-   **VS Code** (optional) or any code editor
    -   Not required, but useful for exploring and modifying the source code.
-   **[Ollama](https://ollama.com/)** installed locally
    -   To run the **Mistral LLM** for answer generation.
		   ```bash
	    ollama pull mistral
	    ```
    -   Verify installation with:
	    ```bash
	    ollama run mistral "Hello
	    ```

<br>

### Clone the Repository
Start by cloning the project repository from GitHub and navigating into it:
```bash
git clone https://github.com/SoubhikSinha/RAG-ElasticSearch-OpenLLM.git
cd RAG-ElasticSearch-OpenLLM
```

<br>

### Environment Setup
It’s recommended to create an isolated Python environment to avoid dependency conflicts:
```bash
conda create --prefix ./rag-elastic python=3.12 -y
conda activate rag-elastic/
```
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

<br>

### Start Elasticsearch + Kibana
Run with Docker - Start a single-node Elasticsearch instance:
```bash
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.1.2

docker run -d \
  --name es-rag \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  --platform=linux/amd64 \
  docker.elastic.co/elasticsearch/elasticsearch:9.1.2
```
Then start Kibana:
```bash
docker pull docker.elastic.co/kibana/kibana:9.1.2

docker run -d \
  --name kibana-rag \
  -p 5601:5601 \
  -e "ELASTICSEARCH_HOSTS=http://es-rag:9200" \
  --link es-rag:es-rag \
  --platform=linux/amd64 \
  docker.elastic.co/kibana/kibana:9.1.2
```

-   Elasticsearch → [http://localhost:9200](http://localhost:9200)
-   Kibana → [http://localhost:5601](http://localhost:5601)

<br>

Verify Installation:<br>
[ ElasticSearch ]
```bash
curl http://localhost:9200/
```
[ Kibana ]
Visit → [http://localhost:5601](http://localhost:5601)

<br>

⚠️ **Note for Mac M1/M2/M3 users:**  
Use `--platform=linux/amd64` (already added above) since **Elastic ML features (ELSER)** are not fully supported on `arm64` images.

<br>

### Run Ingestion
Once Elasticsearch and Kibana are running, execute the following commands to load documents, index them, and test retrieval.
```bash
# 1. Ingest PDFs (download from Google Drive, extract text, split into chunks)
python -m rag.ingestion.py

# 2. Index chunks into Elasticsearch (BM25, ELSER, Dense vectors)
python -m rag.indexing.py

# 3. Run retrieval tests (BM25-only, ELSER-only, Dense-only, Hybrid with RRF)
python -m rag.retrieval.py

# 4. Generate answers using Ollama Mistral (LLM-powered answer generation)
python -m rag.generation.py
```

<br>

### Start API
The backend is powered by **FastAPI**, exposing endpoints for querying, ingestion, and health checks.
<br>
#### 🔹 Run the API server
From the project root, start the FastAPI app with:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
-   `--reload` → auto-restarts the server on code changes
-   API will be available at → [http://localhost:8000](http://localhost:8000)
-   Interactive API docs at → [http://localhost:8000/docs](http://localhost:8000/docs)
<br>

#### 🔹 API Endpoints
-   `POST /query` → Ask a question, get an answer with citations
-   `POST /ingest` → Re-ingest documents from Google Drive
-   `GET /healthz` → Health check
<br>

#### 🔹 Example Requests
**Health check**
```bash
curl -X GET "http://localhost:8000/healthz"
```
**Run ingestion**
```bash
curl -X POST "http://localhost:8000/ingest"
```
**Ask a question**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What does Retrieval-Augmented Generation mean?"}'
```
**Sample Response**
```bash
{
  "answer": "Retrieval-Augmented Generation (RAG) is an approach where a large language model uses external documents retrieved from a search system to ground its responses.",
  "citations": [
    {
      "filename": "rag_paper.pdf",
      "drive_url": "https://drive.google.com/file/xxx",
      "snippet": "Retrieval-Augmented Generation combines..."
    }
  ]
}
```

<br>

### Launch UI
This project includes a **Streamlit web app** for interactive querying.  
It lets you type in questions, toggle retrieval modes, and view answers with citations — all in a simple browser interface.
#### 🔹 Run the Streamlit app
From the project root, start the UI with:
```bash
streamlit run ui/app.py
```
<br>


#### 🔹 Access the UI
Open your browser at → [http://localhost:8501](http://localhost:8501)
You’ll see:
-   **Input box** → Ask any question.
-   **Toggle switch** → Choose retrieval mode (BM25-only, ELSER-only, Dense-only, Hybrid).
-   **Top-k slider** → Adjust how many chunks are retrieved (default = 5).
-   **Answer panel** → Displays grounded response from **Mistral (via Ollama)**.
-   **Citations** → Show filename, snippet, and Google Drive link for each supporting chunk.

<br>

## 📂 Project Structure
The repository is organized as follows:
```bash
RAG-ElasticSearch-OpenLLM/
│
├── data/pdfs/              # Source PDFs (downloaded from Google Drive)
├── rag/                    # Core RAG modules
│   ├── api.py              # FastAPI backend
│   ├── generation.py       # LLM answer generation (Ollama Mistral)
│   ├── guardrails.py       # Query safety filters
│   ├── indexing.py         # Indexing pipeline (BM25, ELSER, Dense vectors)
│   ├── ingestion.py        # Ingestion pipeline (PDFs → text → chunks)
│   ├── retrieval.py        # Retrieval logic (BM25, ELSER, Dense, Hybrid RRF)
│   ├── ui.py               # Streamlit web UI
│   └── __init__.py
│
├── tests/                  # Unit tests
│   ├── test_ingestion.py   # Validate PDF ingestion & chunking
│   ├── test_retrieval.py   # Validate retrieval modes
│   ├── tests_latency.py    # Benchmark search & generation latency
│   └── __init__.py
│
├── venv/                   # Virtual environment (not tracked in git)
├── .env                    # Environment variables (Drive folder, ES URL, Ollama config)
├── .gitignore              # Git ignore rules
├── main.py                 # Entry point (optional orchestration)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

<br>

## 🙏 Acknowledgments
This project was made possible thanks to the contributions of the open-source community and the following tools & resources:
-   **[Elasticsearch](https://www.elastic.co/elasticsearch/?utm_source=chatgpt.com)** → The backbone of hybrid retrieval (BM25, ELSER, dense vectors).
-   **[Kibana](https://www.elastic.co/kibana/?utm_source=chatgpt.com)** → For visualizing, managing ML models, and monitoring pipelines.
-   **[ELSER (Elastic Learned Sparse Encoder)](https://www.elastic.co/guide/en/machine-learning/current/ml-nlp-elser.html?utm_source=chatgpt.com)** → Elastic’s semantic sparse retrieval model.
-   **Sentence Transformers** → For dense embeddings (`all-MiniLM-L6-v2`).
-   **Ollama** → Lightweight local runtime for LLMs on Mac.
-   **Mistral** → Open LLM used for grounded answer generation.
-   **FastAPI** → For building the backend APIs.
-   **Streamlit** → For building an interactive and lightweight UI. 
-   **[PyPDF2](https://pypi.org/project/pypdf2/?utm_source=chatgpt.com)** and **[gdown](https://github.com/wkentaro/gdown?utm_source=chatgpt.com)** → For ingestion of PDFs from Google Drive.
-   **[python-dotenv](https://pypi.org/project/python-dotenv/?utm_source=chatgpt.com)** → For managing environment variables.
-   The **open-source ML community** for inspiring the design of hybrid retrieval pipelines.
-   Special thanks to the **Elastic team** and **Hugging Face community** for their extensive documentation and pre-trained models.
