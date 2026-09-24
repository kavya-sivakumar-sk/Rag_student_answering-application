# RAG-Based Student Notes Question Answering System

A lightweight, fully local Retrieval-Augmented Generation (RAG) system that
lets you upload your study notes (`.txt` or `.pdf`) and ask questions about
them through a Streamlit web interface — **no API key, no PyTorch, no GPU,
no large model downloads.** Just `pip install` and run.

## How it works

1. **Text extraction** — Uploaded `.txt`/`.pdf` files are parsed into raw text.
2. **Chunking** — Text is split into overlapping ~500-character chunks so
   context isn't lost at chunk boundaries.
3. **Vectorizing** — Each chunk is converted into a TF-IDF vector
   (`scikit-learn`), which represents which words matter most in that chunk.
4. **Retrieval** — Your question is vectorized the same way, and cosine
   similarity finds the most relevant chunks (this is the "R" in RAG).
5. **Answering** — The single best-matching sentence across the retrieved
   chunks is extracted and shown as the answer, along with a confidence
   score and the source chunks used (this is the "AG" — augmented
   generation via extraction rather than a full LLM).

This keeps the whole project dependency-light and installs in seconds,
which makes it a good fit for demonstrating the RAG pipeline in a class
assignment without fighting native library / GPU installation issues.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

No models to download — everything runs immediately.

## Running the app

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## Using the app

1. In the sidebar, upload one or more `.txt` or `.pdf` files containing your notes.
   (A sample file is provided at `sample_notes/biology_notes.txt` if you want
   to try it out immediately.)
2. Click **Build / Rebuild Index**.
3. Type a question in the main panel and click **Ask**.
4. The app shows the extracted answer, a confidence score, and the exact
   note chunks it used to find that answer (for transparency/citation).

## Project structure

```
rag-student-notes-qa/
├── app.py               # Streamlit UI
├── rag_engine.py         # Core RAG pipeline (extraction, chunking, TF-IDF retrieval, answer extraction)
├── requirements.txt      # Python dependencies (lightweight, no torch)
├── sample_notes/
│   └── biology_notes.txt # Example notes to test with
└── README.md
```

## Notes & limitations

- This uses **TF-IDF keyword-based retrieval**, not deep-learning
  embeddings — it matches on word overlap rather than semantic meaning.
  It works well for direct, fact-based questions using similar wording to
  the notes, but won't catch paraphrased questions as well as a
  transformer-based embedding model would.
- Answers are **extractive** — a real sentence pulled from your notes,
  not generated free-form prose.
- Scanned/image-only PDFs won't extract text properly since there's no OCR
  step included.

## Upgrading later

If you want closer-to-production semantic search and generative answers
later on, `rag_engine.py` can be swapped to use `sentence-transformers` +
`faiss` for embeddings and a local or API-based LLM for generation — the
overall app structure (`app.py`) stays the same either way.
