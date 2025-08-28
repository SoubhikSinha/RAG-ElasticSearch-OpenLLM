import os
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

import dotenv

dotenv.load_dotenv()
FOLDER_URL = os.getenv("GOOGLE_DRIVE_FOLDER_URL")

class Indexer:
    def __init__(self, index_name="rag_docs", es_url="http://localhost:9200"):
        self.index_name = index_name
        self.es = Elasticsearch(es_url, verify_certs=False)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def create_index(self):
        """
        Creates the Elastic index with mappings for BM25, dense vectors, and ELSER placeholder.
        Compatible with Elasticsearch 9.1.2
        """
        if self.es.indices.exists(index=self.index_name):
            print(f"Index {self.index_name} already exists. Deleting...")
            self.es.indices.delete(index=self.index_name)

        mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "filename": {"type": "keyword"},
                    "drive_url": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "text": {"type": "text"},  # BM25 search
                    "text_expansion": {"type": "rank_features"},  # ELSER placeholder
                    "dense_vector": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        self.es.indices.create(index=self.index_name, body=mapping)
        print(f"✅ Created index: {self.index_name}")

    def index_documents(self, docs):
        """
        Indexes documents with dense embeddings and placeholder ELSER features.
        """
        for doc in docs:
            dense_vector = self.model.encode(doc["text"]).tolist()

            body = {
                "id": doc["id"],
                "filename": doc["filename"],
                "drive_url": doc["drive_url"],
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],

                # In real Elastic Cloud, you'd populate this with ELSER expansion
                "text_expansion": {"dummy_feature": 1.0},

                "dense_vector": dense_vector
            }

            self.es.index(index=self.index_name, id=doc["id"], document=body)

        print(f"✅ Indexed {len(docs)} documents into {self.index_name}")


if __name__ == "__main__":
    from rag.ingestion import process_pdfs

    local_dir = "data/pdfs"   # Assuming PDFs already downloaded manually
    docs = process_pdfs(local_dir, FOLDER_URL)

    indexer = Indexer()
    indexer.create_index()
    indexer.index_documents(docs)
