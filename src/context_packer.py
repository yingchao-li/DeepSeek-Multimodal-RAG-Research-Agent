"""Small, dependency-free helpers for preparing retrieved LLM context."""


def pack_context(chunks, max_tokens):
    if not chunks or max_tokens <= 0:
        return []
    raise NotImplementedError("positive-budget context packing is not implemented yet")
