"""
server.py — Application entry point.

Creates the FastAPI app, registers routes, and starts uvicorn.
All logic lives in the other modules — this file only wires them together.

Usage:
    conda activate rag_server
    python server.py
"""

import uvicorn
from fastapi import FastAPI

from routes import router

app = FastAPI(
    title="Local RAG Knowledge API",
    description="RAG retrieval server. Returns relevant chunks and generates answers via Ollama.",
    version="3.0.0",
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
