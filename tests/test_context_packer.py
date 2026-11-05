"""Offline tests for deterministic context packing."""
from copy import deepcopy


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


def test_normalized_duplicate_keeps_higher_scored_chunk():
    from src.context_packer import pack_context

    chunks = [
        {"text": "  Alpha   BETA  ", "score": 0.2},
        {"text": "alpha beta", "score": 0.9},
        {"text": "unique evidence", "score": 0.5},
    ]

    packed = pack_context(chunks, max_tokens=10, count_tokens=lambda _text: 1)

    assert [chunk["text"] for chunk in packed] == ["alpha beta", "unique evidence"]


def test_near_duplicate_uses_configurable_jaccard_threshold():
    from src.context_packer import pack_context

    chunks = [
        {"text": "red blue green yellow black", "score": 0.9},
        {"text": "red blue green yellow", "score": 0.8},
    ]

    default_packed = pack_context(chunks, max_tokens=10, count_tokens=lambda _text: 1)
    configured_packed = pack_context(
        chunks,
        max_tokens=10,
        count_tokens=lambda _text: 1,
        similarity_threshold=0.8,
    )

    assert len(default_packed) == 2
    assert configured_packed == [chunks[0]]


def test_optional_per_source_cap_increases_context_diversity():
    from src.context_packer import pack_context

    chunks = [
        {"text": "a-high", "score": 0.9, "source": "a.pdf"},
        {"text": "a-second", "score": 0.8, "source": "a.pdf"},
        {"text": "b-evidence", "score": 0.7, "source": "b.pdf"},
    ]

    uncapped = pack_context(chunks, max_tokens=10, count_tokens=lambda _text: 1)
    capped = pack_context(
        chunks,
        max_tokens=10,
        count_tokens=lambda _text: 1,
        max_per_source=1,
    )

    assert [chunk["text"] for chunk in uncapped] == ["a-high", "a-second", "b-evidence"]
    assert [chunk["text"] for chunk in capped] == ["a-high", "b-evidence"]


def test_metadata_is_unchanged_and_score_ties_keep_original_order():
    from src.context_packer import pack_context

    chunks = [
        {
            "text": "figure evidence",
            "score": 0.75,
            "document": "paper-a",
            "source": "a.pdf",
            "page": 4,
            "label": "Figure 3",
            "kind": "figure",
            "chunk_id": "figure-3",
        },
        {
            "text": "table evidence",
            "score": 0.75,
            "document": "paper-b",
            "source": "b.pdf",
            "page": 7,
            "label": "Table 2",
            "kind": "table",
            "chunk_id": "table-2",
        },
    ]
    before = deepcopy(chunks)

    packed = pack_context(chunks, max_tokens=10, count_tokens=lambda _text: 1)

    assert packed == chunks
    assert [chunk["chunk_id"] for chunk in packed] == ["figure-3", "table-2"]
    assert chunks == before
