"""
rag_engine.py
--------------
Lightweight RAG engine for the Student Notes Question Answering System.

Uses TF-IDF + cosine similarity instead of deep-learning embeddings, so it
installs in seconds with no PyTorch/transformers, and needs no GPU, no model
downloads, and no native-library (DLL) dependencies. Good fit for a class
assignment where the goal is to demonstrate the RAG pipeline, not to run a
production-grade language model.

Pipeline:
1. Extract text from uploaded notes (.txt or .pdf)
2. Split text into overlapping chunks
3. Vectorize all chunks with TF-IDF
4. On a question: vectorize the question the same way, rank chunks by
   cosine similarity, and return the best-matching sentence(s) as the answer
"""

import io
import re
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file given as bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from a plain text file given as bytes."""
    return file_bytes.decode("utf-8", errors="ignore")


def load_document(filename: str, file_bytes: bytes) -> str:
    """Dispatch to the right extractor based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Collapse excess whitespace so chunking works on clean text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping character-based chunks, trying to break on
    sentence boundaries so an idea isn't cut in half.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + chunk_size * 0.5:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap if end - overlap > start else end

    return chunks


def split_sentences(text: str) -> List[str]:
    """Very simple sentence splitter used to extract a focused answer."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoteChunk:
    text: str
    source: str
    chunk_id: int


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    chunk_vectors: np.ndarray
    chunks: List[NoteChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(documents: List[Tuple[str, str]]) -> RagIndex:
    """
    Build a TF-IDF index from a list of (filename, raw_text) documents.
    """
    all_chunks: List[NoteChunk] = []

    for filename, raw_text in documents:
        pieces = chunk_text(raw_text)
        for i, piece in enumerate(pieces):
            all_chunks.append(NoteChunk(text=piece, source=filename, chunk_id=i))

    if not all_chunks:
        raise ValueError("No text could be extracted from the uploaded notes.")

    texts = [c.text for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    chunk_vectors = vectorizer.fit_transform(texts)

    return RagIndex(vectorizer=vectorizer, chunk_vectors=chunk_vectors, chunks=all_chunks)


def retrieve(rag_index: RagIndex, query: str, top_k: int = 4) -> List[Tuple[NoteChunk, float]]:
    """Return the top_k most relevant chunks for a query, with similarity scores."""
    query_vec = rag_index.vectorizer.transform([query])
    scores = cosine_similarity(query_vec, rag_index.chunk_vectors)[0]

    ranked_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in ranked_idx:
        if scores[idx] <= 0:
            continue
        results.append((rag_index.chunks[idx], float(scores[idx])))
    return results


# ---------------------------------------------------------------------------
# Answer generation (extractive, via sentence-level TF-IDF match)
# ---------------------------------------------------------------------------

def answer_question(rag_index: RagIndex, question: str, top_k: int = 4) -> dict:
    """
    Retrieve relevant chunks, then pick the single best-matching sentence
    across those chunks as the extracted "answer".
    """
    retrieved = retrieve(rag_index, question, top_k=top_k)
    if not retrieved:
        return {
            "answer": "I couldn't find anything relevant in your notes for that question.",
            "confidence": 0.0,
            "sources": [],
        }

    # Gather all sentences from the retrieved chunks
    candidate_sentences = []
    for chunk, _ in retrieved:
        for sent in split_sentences(chunk.text):
            candidate_sentences.append(sent)

    if not candidate_sentences:
        candidate_sentences = [retrieved[0][0].text]

    # Rank sentences against the question using the same vectorizer vocabulary
    sent_vectorizer = TfidfVectorizer(stop_words="english", vocabulary=rag_index.vectorizer.vocabulary_)
    try:
        sent_vectors = sent_vectorizer.fit_transform(candidate_sentences)
        query_vec = sent_vectorizer.transform([question])
        sent_scores = cosine_similarity(query_vec, sent_vectors)[0]
        best_idx = int(np.argmax(sent_scores))
        best_answer = candidate_sentences[best_idx]
        confidence = float(sent_scores[best_idx])
    except ValueError:
        # Fallback if vocabulary doesn't overlap at all
        best_answer = candidate_sentences[0]
        confidence = 0.0

    sources = [
        {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "similarity": round(score, 3),
        }
        for chunk, score in retrieved
    ]

    return {
        "answer": best_answer,
        "confidence": round(confidence, 3),
        "sources": sources,
    }
