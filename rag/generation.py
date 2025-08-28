import requests
import json
from rag.guardrails import Guardrails


class AnswerGenerator:
    def __init__(self, model_name="mistral"):
        # Ollama runs locally, no Hugging Face pipeline needed
        self.model_name = model_name
        self.guardrails = Guardrails()

    def generate_answer(self, query: str, retrieved_docs):
        # 1. Guardrails
        if not self.guardrails.is_safe_input(query):
            return "❌ Unsafe query refused."
        if self.guardrails.is_prompt_injection(query):
            return "❌ Prompt injection attempt detected and refused."

        if not retrieved_docs:
            return "I don't know."

        # 2. Build context
        context = ""
        for doc, score in retrieved_docs:
            snippet = doc["text"][:400].replace("\n", " ")
            context += f"[{doc['filename']}] {snippet}\n\n"

        # 3. Build chat-style prompt
        prompt = f"""User: {query}
        Context:
        {context}
        Assistant: Answer the question using only the context above. 
        If the answer is not in the context, reply with "I don't know."
        Always cite the source filename(s) in your answer.
        """

        # 4. Call Ollama REST API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": self.model_name, "prompt": prompt}
        )

        if response.status_code != 200:
            return f"⚠️ Ollama error: {response.text}"

        # Ollama streams responses; we only care about final
        output = ""
        for line in response.text.splitlines():
            try:
                data = json.loads(line)
                if "response" in data:
                    output += data["response"]
            except json.JSONDecodeError:
                continue

        print("🔍 RAW MODEL OUTPUT:\n", output)

        answer = output.strip()

        # 5. Guardrails grounding
        if not self.guardrails.is_grounded_output(answer, retrieved_docs):
            return "I don't know."

        # 6. Append citations
        citations = [f"- {doc['filename']} ({doc['drive_url']})" for doc, _ in retrieved_docs]
        return answer + "\n\nCitations:\n" + "\n".join(citations)


# 🔹 Main
if __name__ == "__main__":
    from rag.retrieval import Retriever

    retriever = Retriever()
    generator = AnswerGenerator(model_name="mistral")  # using Ollama mistral

    query = input("\n🔹 Enter your question: ")

    print("\n🔎 Retrieving context...")
    retrieved = retriever.search_hybrid(query, top_k=3)

    if not retrieved:
        print("⚠️ No documents retrieved.")
    else:
        print(f"✅ Retrieved {len(retrieved)} chunks.")

    print("\n🤖 Generating answer...\n")
    answer = generator.generate_answer(query, retrieved)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Final Answer for: {query}\n")
    print(answer)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
