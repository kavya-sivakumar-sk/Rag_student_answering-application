"""
Streamlit app for the RAG-Based Student Notes Question Answering System.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag_engine import build_index, answer_question, load_document

st.set_page_config(page_title="Student Notes Q&A (RAG)", page_icon="📚", layout="wide")

st.title("📚 RAG-Based Student Notes Question Answering System")
st.caption(
    "Upload your notes (.txt or .pdf), then ask questions. "
    "Everything runs locally — no API key needed."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer_dict)

# ---------------------------------------------------------------------------
# Sidebar: upload + build index
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Upload notes")
    uploaded_files = st.file_uploader(
        "Upload one or more files (.txt, .pdf)",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
    )

    top_k = st.slider("Number of chunks to retrieve (top-k)", 1, 8, 4)

    build_clicked = st.button("Build / Rebuild Index", type="primary")

    if build_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            with st.spinner("Reading files and building the TF-IDF index..."):
                documents = []
                for f in uploaded_files:
                    raw_bytes = f.read()
                    text = load_document(f.name, raw_bytes)
                    documents.append((f.name, text))

                st.session_state.rag_index = build_index(documents)
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.history = []

            st.success(f"Indexed {len(uploaded_files)} file(s) into "
                       f"{len(st.session_state.rag_index.chunks)} chunks.")

    if st.session_state.indexed_files:
        st.markdown("**Currently indexed:**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- {name}")

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. Notes are split into overlapping chunks\n"
        "2. Chunks are converted into TF-IDF vectors\n"
        "3. Cosine similarity finds the most relevant chunks for your question\n"
        "4. The best-matching sentence within those chunks is returned as the answer"
    )

# ---------------------------------------------------------------------------
# Main: ask questions
# ---------------------------------------------------------------------------
st.header("2. Ask a question")

if st.session_state.rag_index is None:
    st.info("Upload your notes and click **Build / Rebuild Index** in the sidebar to get started.")
else:
    question = st.text_input("Type your question about the notes:", key="question_input")
    ask_clicked = st.button("Ask")

    if ask_clicked and question.strip():
        with st.spinner("Searching your notes..."):
            result = answer_question(st.session_state.rag_index, question, top_k=top_k)
        st.session_state.history.insert(0, (question, result))

    for q, result in st.session_state.history:
        st.markdown("---")
        st.markdown(f"**Q: {q}**")
        st.markdown(f"### 💡 {result['answer'] if result['answer'] else '_No answer found._'}")
        st.caption(f"Model confidence: {result['confidence']}")

        with st.expander("📖 Sources used for this answer"):
            for src in result["sources"]:
                st.markdown(f"**{src['source']}** — chunk #{src['chunk_id']} "
                            f"(similarity: {src['similarity']})")
                st.write(src["text"])
                st.markdown("")
