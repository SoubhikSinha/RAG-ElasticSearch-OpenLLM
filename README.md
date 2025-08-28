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
- [🔎 Usage](#-usage)
- [📂 Project Structure](#-project-structure)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
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
## 🛠 Tech Stack

## ⚙️ Setup Instructions
### Prerequisites
### Clone the Repository
### Environment Setup
### Start Elasticsearch + Kibana
### Run Ingestion
### Start API
### Launch UI

## 🔎 Usage
## 📂 Project Structure
## 🗺 Roadmap
## 🤝 Contributing
## 📜 License
## 🙏 Acknowledgments
