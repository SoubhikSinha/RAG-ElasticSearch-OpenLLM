import pytest
from rag.retrieval import Retriever

@pytest.fixture(scope="module")
def retriever():
    return Retriever()

def test_search_elser(retriever):
    query = "Explain Docker"
    results = retriever.search_elser(query, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "filename" in results[0][0]

def test_search_hybrid(retriever):
    query = "Explain Docker"
    results = retriever.search_hybrid(query, top_k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "filename" in results[0][0]
