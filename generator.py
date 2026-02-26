"""
generator.py — Answer generation via Ollama.

Takes a question and the retrieved context texts, builds a prompt,
calls the local Ollama model, and returns the generated answer string.

This module is stateless — no model loading, no database connections.
To change the prompt format or swap the LLM: edit this file only.
"""

import ollama

from config import OLLAMA_MODEL


def generate(question: str, context_texts: list[str]) -> str:
    """
    Generate a natural language answer grounded in the provided context.

    Parameters
    ----------
    question      : the user's question
    context_texts : list of raw text strings retrieved from the vector store

    Returns
    -------
    str — the model's generated answer

    How it works:
    1. Join the context chunks into a single context block
    2. Build a prompt that instructs the model to answer from context only
    3. Send to Ollama (blocking call — waits for full response)
    4. Return the stripped answer string

    To change response style (length, tone, format): edit the prompt below.
    """
    context = "\n\n".join(context_texts)

    prompt = f"""You are a helpful assistant. Answer the question using the context below.
Keep your answer direct and try to make it one sentence long. If it's not clear from the context, you may infer but indicate in your response.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.message.content.strip()
