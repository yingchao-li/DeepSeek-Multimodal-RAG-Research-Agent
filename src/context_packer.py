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
                 similarity_threshold=0.9, max_per_source=None):
    if not chunks or max_tokens <= 0:
        return []

    packed = []
    accepted_texts = []
    source_counts = {}
    used_tokens = 0
    ranked = sorted(enumerate(chunks),
                    key=lambda pair: (-float(pair[1].get("score", 0)), pair[0]))
    for _original_index, chunk in ranked:
        if _is_duplicate(chunk.get("text", ""), accepted_texts, similarity_threshold):
            continue
        text = chunk.get("text", "")
        accepted_texts.append((_normalized_text(text), _token_set(text)))
        source = chunk.get("source")
        if max_per_source is not None and source_counts.get(source, 0) >= max_per_source:
            continue
        token_count = max(0, int(count_tokens(chunk.get("text", ""))))
        if used_tokens + token_count > max_tokens:
            continue
        packed.append(chunk)
        used_tokens += token_count
        source_counts[source] = source_counts.get(source, 0) + 1
    return packed


def retrieve_context(retriever, *args, enabled=False, max_tokens=4096,
                     count_tokens=default_count_tokens, similarity_threshold=0.9,
                     max_per_source=None, **kwargs):
    """Run an existing retriever and optionally pack its result list."""
    chunks = retriever(*args, **kwargs)
    if not enabled:
        return chunks
    return pack_context(chunks, max_tokens, count_tokens=count_tokens,
                        similarity_threshold=similarity_threshold,
                        max_per_source=max_per_source)
