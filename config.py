"""
config.py — Single source of truth for all project constants.

Every other module imports from here. To change a model, a path,
or a tuning parameter, this is the only file you need to touch.
"""

from pathlib import Path

# Resolve paths relative to this file so the project works
# regardless of where it is run from.
_BASE = Path(__file__).parent

DOCS_DIR     = _BASE / "docs" / "clash_royale"  # active knowledge base (swap to "elarion" to switch)
CHROMA_DIR   = _BASE / "chroma_db"     # persisted vector database (auto-created by ingest)
COLLECTION   = "game_knowledge"        # name of the ChromaDB collection

EMBED_MODEL  = "all-MiniLM-L6-v2"     # HuggingFace sentence-transformer for embeddings
OLLAMA_MODEL = "llama3.2:3b"           # local Ollama model for generation

TOP_K        = 3                       # number of chunks to retrieve per query
