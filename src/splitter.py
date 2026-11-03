"""
Split long documents into fixed-size text chunks with configurable overlap for embedding and retrieval.
"""


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start += chunk_size - overlap

    return chunks
