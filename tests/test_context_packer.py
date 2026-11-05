"""Offline tests for deterministic context packing."""


def test_empty_input_or_non_positive_budget_returns_empty():
    from src.context_packer import pack_context

    chunk = {"text": "evidence", "score": 1.0}

    assert pack_context([], max_tokens=10) == []
    assert pack_context([chunk], max_tokens=0) == []
    assert pack_context([chunk], max_tokens=-1) == []
