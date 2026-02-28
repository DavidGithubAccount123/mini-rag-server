"""
routes.py — FastAPI route definitions.

All endpoints live here. run.py registers this router but contains
no endpoint logic itself. To add a new endpoint: add it here.
"""

import numpy as np
from fastapi import APIRouter, HTTPException

import core
from config import EMBED_MODEL, OLLAMA_MODEL
from models import AskResponse, QueryRequest, QueryResponse

router = APIRouter()

SIMILARITY_THRESHOLD = 0.7

# Hardcoded semantic cache.
# Keys are example phrasings; values are (answer, sources) tuples.
# Embeddings for each key are computed once on the first request (lazy).
_semantic_cache: dict[str, tuple[str, list]] = {
    "What is your favorite animal, do you like dogs?": (
        "cached answer: Dogs are loyal, loving companions and are considered man's best friend.", []
    ),
    "I am really hungry, what should I eat for dinner tonight?": (
        "cached answer: Pizza is a classic comfort food — a good margherita never disappoints.", []
    ),
    "The weather is terrible outside, what should I do on a rainy day?": (
        "cached answer: Sounds like a good day to stay inside with a warm drink.", []
    ),
}

_cache_key_embeddings: dict[str, list[float]] = {}


def _find_in_cache(query_vec: np.ndarray) -> tuple[str, list] | None:
    """
    Compare a pre-computed query vector to all cached key embeddings.
    Returns (answer, chunks) if the best match scores >= SIMILARITY_THRESHOLD,
    otherwise returns None.
    """
    # Lazily embed any cache keys we haven't processed yet
    for key in _semantic_cache:
        if key not in _cache_key_embeddings:
            _cache_key_embeddings[key] = core.encode_one(key)

    best_score = -1.0
    best_key = None

    for key, emb in _cache_key_embeddings.items():
        emb_vec = np.array(emb)
        score = float(
            np.dot(query_vec, emb_vec)
            / (np.linalg.norm(query_vec) * np.linalg.norm(emb_vec))
        )
        if score > best_score:
            best_score = score
            best_key = key

    if best_score >= SIMILARITY_THRESHOLD and best_key is not None:
        return _semantic_cache[best_key]

    return None


@router.get("/", summary="Health check")
def health_check():
    """Returns server status and how many chunks are currently indexed."""
    return {
        "status":         "ok",
        "chunks_indexed": core.chunk_count(),
        "embed_model":    EMBED_MODEL,
        "ollama_model":   OLLAMA_MODEL,
    }


@router.get("/cache", summary="Inspect the semantic cache")
def get_cache():
    """Returns all current cache entries and how many are loaded."""
    return {
        "count": len(_semantic_cache),
        "entries": [
            {"question": key, "answer": answer}
            for key, (answer, _) in _semantic_cache.items()
        ],
    }


@router.post("/retrieve_query", response_model=QueryResponse, summary="Retrieve relevant chunks")
def query(request: QueryRequest):
    """Embed the question and return the top matching chunks. No LLM generation."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks, _ = core.retrieve(question)
    return QueryResponse(question=question, results=chunks)


@router.post("/rag_query", response_model=AskResponse, summary="Retrieve + generate an answer")
def ask(request: QueryRequest):
    """Retrieve the top matching chunks, then generate a natural language answer."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vec = np.array(core.encode_one(question))

    cached = _find_in_cache(query_vec)
    if cached:
        answer, chunks = cached
        return AskResponse(question=question, answer=answer, sources=chunks)

    chunks, raw_texts = core.retrieve_from_vector(query_vec.tolist())
    answer = core.generate(question, raw_texts)

    return AskResponse(question=question, answer=answer, sources=chunks)
