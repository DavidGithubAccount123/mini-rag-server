"""
store.py — ChromaDB wrapper.

All database logic lives here. Both the ingest pipeline (write)
and the retrieval pipeline (read) import from this module.

To swap the vector database: rewrite this file. Nothing else changes.
"""

import numpy as np
import chromadb

from config import CHROMA_DIR, COLLECTION


# ---------------------------------------------------------------------------
# Connection — established once at import time
# ---------------------------------------------------------------------------

def _connect() -> chromadb.Collection:
    """Open the persistent ChromaDB client and return the collection."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB directory not found at '{CHROMA_DIR}'. "
            "Run 'python ingest.py' first to build the vector database."
        )
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


# ---------------------------------------------------------------------------
# Write side — used by ingest.py
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], embeddings: np.ndarray) -> None:
    """
    Write document chunks and their embeddings to ChromaDB.

    Wipes the existing collection first so reruns always start clean —
    no stale chunks from deleted documents linger in the database.

    Parameters
    ----------
    chunks : list of dicts with keys 'id', 'text', 'source'
    embeddings : numpy array of shape (len(chunks), embedding_dim)
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print("  Cleared existing collection")

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    collection.add(
        ids        = [c["id"]     for c in chunks],
        documents  = [c["text"]   for c in chunks],
        embeddings = embeddings.tolist(),
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

    print(f"  Stored {collection.count()} chunks")


# ---------------------------------------------------------------------------
# Read side — used by retriever.py
# ---------------------------------------------------------------------------

# Collection is loaded once when this module is first imported by the server.
# We use a lazy initialisation pattern so ingest.py (which writes before
# the collection exists) can also import from this module safely.
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    """Return the collection, connecting on first call."""
    global _collection
    if _collection is None:
        _collection = _connect()
    return _collection


def query_chunks(embedding: list[float], top_k: int) -> dict:
    """
    Search the vector store for the top_k most similar chunks.

    Parameters
    ----------
    embedding : query vector as a plain Python list of floats
    top_k     : number of results to return

    Returns
    -------
    Raw ChromaDB result dict with keys: documents, metadatas, distances
    """
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def chunk_count() -> int:
    """Return the number of chunks currently stored in the collection."""
    return get_collection().count()
