"""
retriever.py — Retrieval pipeline.

Given a question string, embeds it and fetches the most relevant
chunks from the vector store. Returns both structured objects
(for the API response) and raw text strings (for the generator prompt).
"""

import embedder
import store
from config import TOP_K
from models import RetrievedChunk


def retrieve(question: str) -> tuple[list[RetrievedChunk], list[str]]:
    """
    Embed the question and return the top K most relevant chunks.

    Parameters
    ----------
    question : the user's question as a plain string

    Returns
    -------
    chunks    : list[RetrievedChunk] — structured results for the API response
    raw_texts : list[str]           — plain text strings for building the LLM prompt

    How it works:
    1. embed the question into a vector using embedder.encode_one()
    2. query the vector store for the TOP_K nearest vectors
    3. convert raw ChromaDB results into RetrievedChunk objects
    4. ChromaDB returns cosine distance (0 = identical, 2 = opposite)
       so we convert: similarity = 1 - distance
    """
    query_vector = embedder.encode_one(question)
    results = store.query_chunks(query_vector, TOP_K)

    # ChromaDB returns lists-of-lists for batch support.
    # We only send one query so we unwrap the outer list with [0].
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
