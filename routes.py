"""
routes.py — FastAPI route definitions.

All endpoints live here. run.py registers this router but contains
no endpoint logic itself. To add a new endpoint: add it here.
"""

from fastapi import APIRouter, HTTPException

import core
from config import EMBED_MODEL, OLLAMA_MODEL
from models import AskResponse, QueryRequest, QueryResponse

router = APIRouter()


@router.get("/", summary="Health check")
def health_check():
    """Returns server status and how many chunks are currently indexed."""
    return {
        "status":         "ok",
        "chunks_indexed": core.chunk_count(),
        "embed_model":    EMBED_MODEL,
        "ollama_model":   OLLAMA_MODEL,
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

    chunks, raw_texts = core.retrieve(question)
    answer = core.generate(question, raw_texts)

    return AskResponse(question=question, answer=answer, sources=chunks)
