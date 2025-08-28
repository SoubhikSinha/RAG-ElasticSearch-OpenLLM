import os
import time
import pytest

from rag.ingestion import download_pdfs_from_gdrive, process_pdfs
from rag.retrieval import Retriever


FOLDER_URL = os.getenv("GOOGLE_DRIVE_FOLDER_URL")


@pytest.mark.ingestion
def test_ingestion_latency():
    """
    Ensure ingestion (download + processing) finishes within 3s 
    for a small dataset.
    """
    start = time.time()
    local_dir = download_pdfs_from_gdrive(FOLDER_URL)
    docs = process_pdfs(local_dir, FOLDER_URL)
    elapsed = time.time() - start

    assert len(docs) > 0, "No docs were ingested"
    assert elapsed < 3, f"Ingestion took too long: {elapsed:.2f}s"
    print(f"✅ Ingestion latency: {elapsed:.2f}s")


@pytest.mark.retrieval
def test_retrieval_latency():
    """
    Ensure retrieval finishes within 3s.
    """
    retriever = Retriever()
    query = "Explain Docker"

    start = time.time()
    results = retriever.search_hybrid(query, top_k=3)
    elapsed = time.time() - start

    assert len(results) > 0, "No results retrieved"
    assert elapsed < 3, f"Retrieval took too long: {elapsed:.2f}s"
    print(f"✅ Retrieval latency: {elapsed:.2f}s")
