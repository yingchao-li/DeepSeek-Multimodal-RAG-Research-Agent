"""
Web fallback using DuckDuckGo text search: return short snippets as pseudo-documents ([web]-tagged sources) for RAG context.
"""

from __future__ import annotations


def fetch_web_snippets(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns list of dicts: {source, text, score} suitable for RAG context.
    `source` is prefixed with [web] for UI.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        from ddgs import DDGS
    except ImportError:
        return []

    out: list[dict] = []
    try:
        with DDGS(timeout=15) as ddgs:
            for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()
                href = (r.get("href") or "").strip()
                if not body and not title:
                    continue
                label = title or href or f"result_{i + 1}"
                text = body if body else title
                out.append({
                    "source": f"[web] {label}",
                    "text": text[:2000],
                    "score": 0.0,
                    "url": href if href.startswith(('https://', 'http://')) else None,
                })
    except Exception:
        return out

    return out
