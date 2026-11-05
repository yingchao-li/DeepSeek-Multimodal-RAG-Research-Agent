"""Offline tests for deterministic context packing."""


def test_empty_input_or_non_positive_budget_returns_empty():
    from src.context_packer import pack_context

    chunk = {"text": "evidence", "score": 1.0}

    assert pack_context([], max_tokens=10) == []
    assert pack_context([chunk], max_tokens=0) == []
    assert pack_context([chunk], max_tokens=-1) == []


def test_score_order_and_greedy_budget_skip_chunks_that_do_not_fit():
    from src.context_packer import pack_context

    chunks = [
        {"text": "small", "score": 0.7},
        {"text": "largest-score", "score": 0.9},
        {"text": "middle-score", "score": 0.8},
    ]
    token_sizes = {"small": 2, "largest-score": 4, "middle-score": 3}

    packed = pack_context(
        chunks,
        max_tokens=6,
        count_tokens=lambda text: token_sizes[text],
    )

    assert [chunk["text"] for chunk in packed] == ["largest-score", "small"]
    assert sum(token_sizes[chunk["text"]] for chunk in packed) <= 6


def test_default_token_counter_is_dependency_free_word_count():
    from src.context_packer import pack_context

    chunks = [
        {"text": "one two three", "score": 1.0},
        {"text": "four", "score": 0.5},
    ]

    assert pack_context(chunks, max_tokens=3) == [chunks[0]]
