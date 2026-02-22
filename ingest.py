"""
ingest.py — Build the vector database from the docs/ folder.

Run this ONCE before starting the server, and re-run it any time
you change or add documents.

What this script does:
  1. Reads every .txt file in docs/
  2. Splits each file into chunks (one paragraph = one chunk)
  3. Loads the all-MiniLM-L6-v2 embedding model from HuggingFace
  4. Converts every chunk into a 384-dimensional vector
  5. Stores (vector, text, source filename) in ChromaDB on disk

After this runs, ./chroma_db/ will exist on disk and the server
can load it instantly without re-embedding anything.
"""

import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# --- Config ---
DOCS_DIR   = Path(__file__).parent / "docs_cr"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "game_knowledge"

# The embedding model. all-MiniLM-L6-v2 is small (80 MB), fast,
# and works well for semantic similarity tasks.
# First run will download it from HuggingFace (~80 MB).
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_chunks(docs_dir: Path) -> list[dict]:
    """
    Read all .txt files and split them into paragraph-level chunks.

    Each chunk is a dict:
        { "text": "...", "source": "characters.txt", "id": "characters.txt::0" }

    Splitting on double newlines (blank lines) gives us one logical
    section per chunk — a good granularity for game-wiki style content.
    """
    chunks = []
    for txt_file in sorted(docs_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")

        # Split on one or more blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            chunks.append({
                "text":   para,
                "source": txt_file.name,
                "id":     f"{txt_file.name}::{i}",
            })
        print(f"  Loaded {len(paragraphs)} chunks from {txt_file.name}")

    return chunks


def main():
    print("=== RAG Ingestion: Elarion: Shattered Realms ===\n")

    # 1. Load documents
    print("Loading documents...")
    chunks = load_chunks(DOCS_DIR)
    print(f"  Total chunks: {len(chunks)}\n")

    # 2. Load embedding model
    # SentenceTransformer downloads the model on first run and caches it locally.
    print(f"Loading embedding model '{EMBED_MODEL}'...")
    print("  (First run downloads ~80 MB from HuggingFace — subsequent runs are instant)\n")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. Embed all chunks in one batch
    # model.encode() takes a list of strings and returns a numpy array
    # of shape (num_chunks, 384). Each row is one embedding vector.
    print("Embedding chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"  Embedded {len(embeddings)} chunks into {embeddings.shape[1]}-dimensional vectors\n")

    # 4. Store in ChromaDB
    # PersistentClient saves to disk at the given path.
    # We recreate the collection from scratch each time ingest runs
    # so stale data from deleted docs doesn't linger.
    print(f"Storing in ChromaDB at '{CHROMA_DIR}'...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete and recreate the collection so reruns start clean
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print("  Cleared existing collection")

    collection = client.create_collection(
        name=COLLECTION,
        # cosine distance = 1 - cosine_similarity
        # ChromaDB returns distances, so lower = more similar
        metadata={"hnsw:space": "cosine"},
    )

    # ChromaDB expects embeddings as a list of lists (not numpy array)
    collection.add(
        ids        = [c["id"]     for c in chunks],
        documents  = [c["text"]   for c in chunks],
        embeddings = embeddings.tolist(),
        metadatas  = [{"source": c["source"]} for c in chunks],
    )

    print(f"  Stored {collection.count()} chunks\n")
    print("=== Ingestion complete. You can now run: python server.py ===")


if __name__ == "__main__":
    main()
