"""
models.py — Pydantic request and response schemas.

Defines the shape of data coming into and going out of the API.
Imported by routes.py and retriever.py.
"""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """Incoming request body for /query and /ask endpoints."""
    question: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What are the P.E.K.K.A's weaknesses?"},
                {"question": "What is the relationship between the Giant and the Witch?"},
                {"question": "Who counters the Hog Rider?"},
            ]
        }
    }


class RetrievedChunk(BaseModel):
    """A single chunk returned from the vector store."""
    text:   str
    source: str   # filename the chunk came from
    score:  float # cosine similarity score (higher = more relevant)


class QueryResponse(BaseModel):
    """Response from /query — raw retrieved chunks, no generation."""
    question: str
    results:  list[RetrievedChunk]


class AskResponse(BaseModel):
    """Response from /ask — generated answer + the chunks used to produce it."""
    question: str
    answer:   str
    sources:  list[RetrievedChunk]
