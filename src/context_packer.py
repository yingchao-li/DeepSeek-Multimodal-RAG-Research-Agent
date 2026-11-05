"""Small, dependency-free helpers for preparing retrieved LLM context."""


def default_count_tokens(text):
    return len(str(text or "").split())


def pack_context(chunks, max_tokens, *, count_tokens=default_count_tokens):
    if not chunks or max_tokens <= 0:
        return []

    packed = []
    used_tokens = 0
    for chunk in sorted(chunks, key=lambda item: float(item.get("score", 0)), reverse=True):
        token_count = max(0, int(count_tokens(chunk.get("text", ""))))
        if used_tokens + token_count > max_tokens:
            continue
        packed.append({key: chunk[key] for key in ("text", "score") if key in chunk})
        used_tokens += token_count
    return packed
