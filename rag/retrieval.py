from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index_name="rag_docs", es_url="http://localhost:9200"):
        self.index_name = index_name
        self.es = Elasticsearch(es_url, verify_certs=False)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search_bm25(self, query, top_k=5):
        """
        Classic BM25 keyword search.
        """
        res = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {"text": query}
                },
                "size": top_k
            }
        )
        return [(hit["_source"], hit["_score"]) for hit in res["hits"]["hits"]]

    def search_dense(self, query, top_k=5):
        """
        Dense vector search using cosine similarity.
        """
        query_vector = self.model.encode(query).tolist()
        res = self.es.search(
            index=self.index_name,
            knn={
                "field": "dense_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": 50
            }
        )
        return [(hit["_source"], hit["_score"]) for hit in res["hits"]["hits"]]

    def search_elser(self, query, top_k=5):
        """
        Stub for ELSER sparse search.
        In real Elastic Cloud, you'd expand query using ELSER model.
        Here, we simulate by treating it as another BM25 field.
        """
        res = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {"text": query}
                },
                "size": top_k
            }
        )
        return [(hit["_source"], hit["_score"]) for hit in res["hits"]["hits"]]

    def reciprocal_rank_fusion(self, results_list, top_k=5, k=60):
        """
        Combine multiple result sets using Reciprocal Rank Fusion (RRF).
        results_list: list of lists of (doc, score) from different retrievers
        """
        scores = {}
        for results in results_list:
            for rank, (doc, _) in enumerate(results):
                doc_id = doc["id"]
                rr_score = 1.0 / (k + rank + 1)
                scores[doc_id] = scores.get(doc_id, 0) + rr_score

        # sort docs by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_docs = []
        for doc_id, score in ranked[:top_k]:
            # get full document from ES
            doc = self.es.get(index=self.index_name, id=doc_id)["_source"]
            fused_docs.append((doc, score))
        return fused_docs

    def search_hybrid(self, query, top_k=5):
        """
        Hybrid search: BM25 + Dense + ELSER stub fused with RRF.
        """
        bm25 = self.search_bm25(query, top_k=top_k)
        dense = self.search_dense(query, top_k=top_k)
        elser = self.search_elser(query, top_k=top_k)

        return self.reciprocal_rank_fusion([bm25, dense, elser], top_k=top_k)


if __name__ == "__main__":
    retriever = Retriever()

    query = "What is Docker?"
    print("\n🔹 BM25:")
    print(retriever.search_bm25(query))

    print("\n🔹 Dense:")
    print(retriever.search_dense(query))

    print("\n🔹 ELSER (stub):")
    print(retriever.search_elser(query))

    print("\n🔹 Hybrid (RRF):")
    print(retriever.search_hybrid(query))
