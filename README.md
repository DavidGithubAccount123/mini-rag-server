# RAG Server

A local retrieval-augmented generation server. Ask questions, get answers grounded in your own documents. Runs fully offline — no API keys required.

**Stack:** FastAPI · ChromaDB · sentence-transformers (all-MiniLM-L6-v2) · Ollama (llama3.2:3b)

---

## Prerequisites

Install these once on your machine before anything else.

**1. [Anaconda](https://www.anaconda.com/download)** — Python environment manager

**2. [Ollama](https://ollama.com)** — runs the local LLM

After installing Ollama, pull the model (one-time, ~2 GB download):
```bash
ollama pull llama3.2:3b
```

---

## Setup

**1. Create a conda environment**
```bash
conda create -n rag_server python=3.11 -y
conda activate rag_server
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Running

**First time only — build the vector database:**
```bash
conda activate rag_server
cd path/to/rag_server
python ingest.py
```

**Start the server:**
```bash
python run.py
```

Server runs at `http://localhost:8000`. To stop it: `Ctrl + C`.

---

## Using the API

Open `http://localhost:8000/docs` in your browser for an interactive UI.

| Endpoint | What it does |
|----------|-------------|
| `GET /` | Health check |
| `POST /query` | Returns raw retrieved chunks, no generation |
| `POST /ask` | Returns a generated answer + source chunks |

Both `POST` endpoints take:
```json
{ "question": "Your question here" }
```

---

## Project structure

```
rag_server/
    config.py       ← all settings (models, paths, TOP_K)
    chunker.py      ← text chunking strategies
    embedder.py     ← embedding model
    store.py        ← ChromaDB read/write
    retriever.py    ← retrieval pipeline
    generator.py    ← prompt + Ollama call
    models.py       ← API schemas
    routes.py       ← API endpoints
    run.py          ← entry point
    ingest.py       ← builds the vector database
    docs/
        clash_royale/   ← active knowledge base
        elarion/        ← inactive
```

---

## Changing the knowledge base

1. Add `.txt` files to a folder inside `docs/`
2. Update `DOCS_DIR` in `config.py` to point at the new folder
3. Re-run `python ingest.py`

To swap chunking strategy, change one line in `chunker.py`:
```python
chunk = by_paragraph   # or by_fixed_size, by_sentence
```
Then re-run `python ingest.py`.
