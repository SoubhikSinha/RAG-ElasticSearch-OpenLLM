import sys, os
import streamlit as st

# ✅ Ensure parent path is added before imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retrieval import Retriever
from rag.generation import AnswerGenerator

retriever = Retriever()
generator = AnswerGenerator(model_name="mistral")

# 🎨 Page config
st.set_page_config(
    page_title="RAG System Demo",
    page_icon="📘",
    layout="wide",
)

# 🌟 Header
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86C1;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1rem;
        text-align: center;
        color: #5D6D7E;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #2E86C1;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #1B4F72;
    }
    .card {
        background-color: #F8F9F9;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .citation {
        font-size: 0.9rem;
        color: #566573;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📘 RAG System Demo</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions over your documents with Retrieval-Augmented Generation</div>', unsafe_allow_html=True)

# 🔍 Input section
query = st.text_input("💬 Enter your question:", placeholder="e.g., Explain Docker")
col1, col2 = st.columns([1,1])
with col1:
    mode = st.radio("Retrieval Mode:", ["Hybrid", "ELSER", "BM25", "Dense"])
with col2:
    top_k = st.slider("Top-k results:", 1, 10, 5)

# 🚀 Ask button
if st.button("🔎 Ask"):
    with st.spinner("Retrieving and generating answer..."):
        mode = mode.lower()
        if mode == "hybrid":
            retrieved = retriever.search_hybrid(query, top_k=top_k)
        elif mode == "elser":
            retrieved = retriever.search_elser(query, top_k=top_k)
        elif mode == "bm25":
            retrieved = retriever.search_bm25(query, top_k=top_k)
        elif mode == "dense":
            retrieved = retriever.search_dense(query, top_k=top_k)
        else:
            retrieved = []

        answer = generator.generate_answer(query, retrieved)

        # 🟦 Answer Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✅ Answer")
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # 📑 Citations Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📑 Citations")
        for doc, score in retrieved:
            st.markdown(
                f"""
                - **[{doc['filename']}]({doc['drive_url']})**  
                <div class="citation">_Snippet:_ {doc['text'][:200]}...</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)
