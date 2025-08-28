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

## Why this Project ?
-   🔍 **Search relevance**: Elastic’s **ELSER** bridges BM25 and dense retrieval for higher recall.
-   🧠 **Grounded generation**: Answers are built strictly from retrieved evidence — no hallucinations.
-   ⚡ **Scalable infra**: Elasticsearch 9.1.2 + Kibana + ML nodes, all containerized with Docker.
-   🎯 **End-to-end example**: From ingestion → indexing → retrieval → generation → UI.
