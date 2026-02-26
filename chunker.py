"""
chunker.py — Pluggable text chunking strategies.

To change how documents are split into chunks, edit the last line:
    chunk = by_paragraph      ← current strategy
    chunk = by_fixed_size     ← swap to this (or any other function below)

Every strategy takes a raw text string and returns a list of chunk strings.
ingest.py calls chunker.chunk() — it never needs to know which strategy
is active.
"""


def by_paragraph(text: str) -> list[str]:
    """
    Split on blank lines (double newlines).

    Best for: documents that are already structured into logical sections,
    like our game wiki files where each character is its own paragraph.
    One blank line between sections = one chunk per section.
    """
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def by_fixed_size(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split into fixed-size character chunks with optional overlap.

    Best for: unstructured text blobs (PDFs, articles, reports) where
    there are no natural section boundaries to split on.

    Overlap ensures that context at chunk boundaries is not lost — the
    last `overlap` characters of one chunk become the first characters
    of the next.

    Parameters
    ----------
    text    : raw text to split
    size    : target chunk size in characters
    overlap : how many characters to repeat between adjacent chunks
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if c]


def by_sentence(text: str) -> list[str]:
    """
    Split on sentence boundaries (period + space or newline).

    Best for: narrative text where paragraph breaks are inconsistent
    but sentences are well-formed. Less precise than a proper NLP
    sentence tokeniser but requires no additional dependencies.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# Active strategy — change this one line to swap chunking behaviour.
# ingest.py calls chunker.chunk(text) and is unaffected by which
# function is assigned here.
# ---------------------------------------------------------------------------

chunk = by_paragraph
