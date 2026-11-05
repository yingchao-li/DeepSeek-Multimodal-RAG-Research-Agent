"""Small, dependency-free helpers for preparing retrieved LLM context."""
import re


def default_count_tokens(text):
    return len(str(text or "").split())


def _normalized_text(text):
    return " ".join(str(text or "").casefold().split())


def _token_set(text):
    return set(re.findall(r"\w+", _normalized_text(text), flags=re.UNICODE))


def _is_duplicate(text, accepted_texts, threshold):
    normalized = _normalized_text(text)
    tokens = _token_set(text)
    for accepted_normalized, accepted_tokens in accepted_texts:
        if normalized == accepted_normalized:
            return True
        union = tokens | accepted_tokens
        if union and len(tokens & accepted_tokens) / len(union) >= threshold:
            return True
    return False


def pack_context(chunks, max_tokens, *, count_tokens=default_count_tokens,
                 similarity_threshold=0.9):
    if not chunks or max_tokens <= 0:
        return []

    packed = []
    accepted_texts = []
    used_tokens = 0
    for chunk in sorted(chunks, key=lambda item: float(item.get("score", 0)), reverse=True):
        if _is_duplicate(chunk.get("text", ""), accepted_texts, similarity_threshold):
            continue
        text = chunk.get("text", "")
        accepted_texts.append((_normalized_text(text), _token_set(text)))
        token_count = max(0, int(count_tokens(chunk.get("text", ""))))
        if used_tokens + token_count > max_tokens:
            continue
        packed.append({key: chunk[key] for key in ("text", "score") if key in chunk})
        used_tokens += token_count
    return packed
