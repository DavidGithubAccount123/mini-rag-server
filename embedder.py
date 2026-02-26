"""
embedder.py — Embedding model wrapper.

Loads the SentenceTransformer model once at import time and exposes
two simple functions. Both ingest.py and retriever.py import from here,
so the model is defined and configured in exactly one place.

To swap the embedding model: change EMBED_MODEL in config.py.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBED_MODEL

print(f"Loading embedding model '{EMBED_MODEL}'...")
_model = SentenceTransformer(EMBED_MODEL)
print("  Embedding model ready.")


def encode(texts: list[str]) -> np.ndarray:
    """
    Embed a list of strings into a 2D numpy array.

    Parameters
    ----------
    texts : list of strings to embed

    Returns
    -------
    np.ndarray of shape (len(texts), embedding_dim)
    Each row is the embedding vector for the corresponding text.

    Used by ingest.py to embed all document chunks in one batch.
    """
    return _model.encode(texts, show_progress_bar=True)


def encode_one(text: str) -> list[float]:
    """
    Embed a single string and return it as a plain Python list.

    Returns a list (not numpy array) because ChromaDB's query()
    expects list[float] for query_embeddings.

    Used by retriever.py to embed the user's question at query time.
    """
    return _model.encode(text).tolist()
