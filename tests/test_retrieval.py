from rag.retrieval import Retriever

def test_retrieval():
    retriever = Retriever()

    query = "Explain Docker"

    print("\n--- 🔹 ELSER-only ---")
    results_elser = retriever.search(query, top_k=5, mode="elser")
    for doc, score in results_elser:
        print(f"- {doc['filename']} (score={score:.4f})")

    print("\n--- 🔹 Hybrid (BM25 + Dense + ELSER + RRF) ---")
    results_hybrid = retriever.search(query, top_k=5, mode="hybrid")
    for doc, score in results_hybrid:
        print(f"- {doc['filename']} (score={score:.4f})")


if __name__ == "__main__":
    test_retrieval()
