import re
from sentence_transformers import SentenceTransformer, util


class Guardrails:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Blocklist patterns (unsafe/harmful)
        self.block_patterns = [
            r"how to make.*bomb",
            r"kill myself",
            r"suicide",
            r"terrorism",
            r"child abuse",
            r"nuke",
        ]

        # Prompt injection patterns
        self.injection_patterns = [
            r"ignore (all|previous|above) instructions",
            r"forget (all|previous) instructions",
            r"reveal (system|hidden) prompt",
            r"you are now",
            r"act as",
            r"pretend to be",
            r"jailbreak",
            r"dan",
        ]

        # Load embedding model once
        self.embedder = SentenceTransformer(embedding_model)

    def is_safe_input(self, query: str) -> bool:
        """
        Keyword-based check for unsafe queries.
        """
        for pattern in self.block_patterns:
            if re.search(pattern, query.lower()):
                return False
        return True

    def is_prompt_injection(self, query: str) -> bool:
        """
        Detect common prompt injection / jailbreak attempts.
        """
        for pattern in self.injection_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False

    def is_grounded_output(self, answer: str, retrieved_docs, threshold: float = 0.4) -> bool:
        if "i don't know" in answer.lower():
            return True

        contexts = [doc["text"] for doc, _ in retrieved_docs]

        # Embeddings
        ans_emb = self.embedder.encode(answer, convert_to_tensor=True)
        ctx_embs = self.embedder.encode(contexts, convert_to_tensor=True)
        sims = util.cos_sim(ans_emb, ctx_embs)
        max_sim = sims.max().item()

        # Word overlap (backup)
        context_text = " ".join(contexts).lower()
        overlap = sum(1 for word in answer.lower().split() if word in context_text)

        return max_sim >= threshold or overlap > 5

