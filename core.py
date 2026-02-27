"""
core.py — All ML/AI logic for the RAG pipeline.

This is the service layer. ingest.py and routes.py call functions
from here — neither pipeline contains ML logic of its own.

Sections
--------
CHUNKING   — text splitting strategies (swap one line to change strategy)
EMBEDDING  — sentence-transformer wrapper
STORE      — ChromaDB read/write
RETRIEVAL  — embed a question + query the store
GENERATION — prompt building + Ollama call
"""

import re

import chromadb
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION, EMBED_MODEL, OLLAMA_MODEL, TOP_K
from models import RetrievedChunk


# ---------------------------------------------------------------------------
# CHUNKING — text splitting strategies
# To change strategy: swap the last line.  chunk = by_fixed_size, etc.
# ---------------------------------------------------------------------------

def by_paragraph(text: str) -> list[str]:
    """Split on blank lines. Best for structured docs with clear sections."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def by_fixed_size(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """Split into fixed-size character chunks with overlap. Best for unstructured blobs."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def by_sentence(text: str) -> list[str]:
    """Split on sentence boundaries. Best for narrative text."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# Active chunking strategy — change this one line to swap.
chunk = by_paragraph


# ---------------------------------------------------------------------------
# EMBEDDING — SentenceTransformer loaded once at import time
# ---------------------------------------------------------------------------

print(f"Loading embedding model '{EMBED_MODEL}'...")
_model = SentenceTransformer(EMBED_MODEL)
print("  Embedding model ready.")


def encode(texts: list[str]) -> np.ndarray:
    """Embed a list of strings. Returns (n, dim) numpy array. Used by ingest."""
    return _model.encode(texts, show_progress_bar=True)


def encode_one(text: str) -> list[float]:
    """Embed a single string. Returns a plain list. Used by retrieval."""
    return _model.encode(text).tolist()


# ---------------------------------------------------------------------------
# STORE — ChromaDB read/write
# ---------------------------------------------------------------------------

_collection: chromadb.Collection | None = None


def _connect() -> chromadb.Collection:
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB directory not found at '{CHROMA_DIR}'. "
            "Run 'python ingest.py' first to build the vector database."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


def get_collection() -> chromadb.Collection:
    """Return the collection, connecting on first call (lazy init)."""
    global _collection
    if _collection is None:
        _collection = _connect()
    return _collection


def save_chunks(chunks: list[dict], embeddings: np.ndarray) -> None:
    """Write chunks and embeddings to ChromaDB. Wipes existing collection first."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        client.delete_collection(COLLECTION)
        print("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids        = [c["id"]     for c in chunks],
        documents  = [c["text"]   for c in chunks],
        embeddings = embeddings.tolist(),
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

    print(f"Stored {collection.count()} chunks")


def query_chunks(embedding: list[float], top_k: int) -> dict:
    """Search the vector store for the top_k most similar chunks."""
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def chunk_count() -> int:
    """Return the number of chunks currently stored."""
    return get_collection().count()


# ---------------------------------------------------------------------------
# RETRIEVAL — embed a question and fetch matching chunks
# ---------------------------------------------------------------------------

def retrieve_from_vector(query_vector: list[float]) -> tuple[list[RetrievedChunk], list[str]]:
    """
    Fetch the top K most relevant chunks using a pre-computed query vector.
    Use this when you have already embedded the question and don't want to embed again.
    """
    results = query_chunks(query_vector, TOP_K)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    chunks = [
        RetrievedChunk(
            text   = doc,
            source = meta["source"],
            score  = round(1 - dist, 4),
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]

    return chunks, documents


def retrieve(question: str) -> tuple[list[RetrievedChunk], list[str]]:
    """Embed the question then retrieve. Convenience wrapper around retrieve_from_vector."""
    return retrieve_from_vector(encode_one(question))


# ---------------------------------------------------------------------------
# GENERATION — prompt building + Ollama call
# ---------------------------------------------------------------------------

def generate(question: str, context_texts: list[str]) -> str:
    """Build a prompt from context and return the Ollama model's answer."""
    context = "\n\n".join(context_texts)

    prompt = f"""You are a helpful assistant. Answer the question using the context below.
Keep your answer direct and try to make it one sentence long. If it's not clear from the context, you may infer but indicate in your response.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.message.content.strip()
