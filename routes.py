"""
routes.py — FastAPI route definitions.

All endpoint logic lives here. server.py registers this router
but contains no endpoint logic itself.

To add a new endpoint: add it here. server.py never needs to change.
"""

from fastapi import APIRouter, HTTPException

import generator
import retriever
import store
from config import EMBED_MODEL, OLLAMA_MODEL
from models import AskResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/", summary="Health check")
def health_check():
    """Returns server status and how many chunks are currently indexed."""
    return {
        "status":         "ok",
        "chunks_indexed": store.chunk_count(),
        "embed_model":    EMBED_MODEL,
        "ollama_model":   OLLAMA_MODEL,
    }


@router.post("/query", response_model=QueryResponse, summary="Retrieve relevant chunks")
def query(request: QueryRequest):
    """
    Embed the question and return the top matching chunks.
    No LLM generation — raw retrieval only.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks, _ = retriever.retrieve(question)
    return QueryResponse(question=question, results=chunks)


@router.post("/ask", response_model=AskResponse, summary="Retrieve + generate an answer")
def ask(request: QueryRequest):
    """
    Retrieve the top matching chunks, then generate a natural language
    answer using the local Ollama model.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks, raw_texts = retriever.retrieve(question)
    answer = generator.generate(question, raw_texts)

    return AskResponse(question=question, answer=answer, sources=chunks)
