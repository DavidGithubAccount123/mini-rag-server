"""
ingest.py — Build the vector database from the docs folder.

Run this ONCE before starting the server, and re-run it any time
you add, remove, or change documents.

Usage:
    conda activate rag_server
    python ingest.py

This file is intentionally thin — it orchestrates the pipeline
but contains no logic of its own. To change behaviour:
    - Chunking strategy  → edit chunker.py
    - Embedding model    → edit config.py (EMBED_MODEL)
    - Vector store       → edit store.py
    - Docs folder        → edit config.py (DOCS_DIR)
"""

from pathlib import Path

import chunker
import embedder
import store
from config import DOCS_DIR


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt files from docs_dir and split them into chunks.

    Returns a list of dicts:
        { "id": "cards.txt::0", "text": "...", "source": "cards.txt" }
    """
    all_chunks = []

    for txt_file in sorted(docs_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        chunks = chunker.chunk(text)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "id":     f"{txt_file.name}::{i}",
                "text":   chunk_text,
                "source": txt_file.name,
            })

        print(f"  {txt_file.name}: {len(chunks)} chunks")

    return all_chunks


def main():
    print("=== RAG Ingest Pipeline ===\n")

    # 1. Load and chunk documents
    print(f"Loading documents from '{DOCS_DIR}'...")
    chunks = load_documents(DOCS_DIR)
    print(f"  Total chunks: {len(chunks)}\n")

    # 2. Embed all chunks
    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    print(f"  Embedded {len(embeddings)} chunks "
          f"into {embeddings.shape[1]}-dimensional vectors\n")

    # 3. Save to vector store
    print("Saving to vector store...")
    store.save_chunks(chunks, embeddings)
    print()
    print("=== Ingestion complete. You can now run: python server.py ===")


if __name__ == "__main__":
    main()
