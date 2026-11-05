# Context packer TDD log

The feature was implemented as six ordered red/green cycles. Every red commit contains tests only for that cycle; every green commit follows a targeted failing run with a passing full offline test suite. The paid integration test remained skipped throughout.

| Cycle | Behavior | Failing test(s) | Red failure excerpt | Red commit | Green commit |
|---:|---|---|---|---|---|
| 1 | Empty chunks or a non-positive budget return `[]`. | `test_empty_input_or_non_positive_budget_returns_empty` | `ModuleNotFoundError: No module named 'src.context_packer'` | `04fa18f` | `55ed67f` |
| 2 | Score-descending greedy packing skips a chunk that does not fit and continues; token counting is injectable with a dependency-free default. | `test_score_order_and_greedy_budget_skip_chunks_that_do_not_fit`; `test_default_token_counter_is_dependency_free_word_count` | `unexpected keyword argument 'count_tokens'`; `NotImplementedError: positive-budget context packing...` | `bdfbc1f` | `d437c86` |
| 3 | Normalized duplicates and token-set Jaccard near-duplicates are removed; the threshold defaults to `0.9` and is configurable. | `test_normalized_duplicate_keeps_higher_scored_chunk`; `test_near_duplicate_uses_configurable_jaccard_threshold` | duplicate remained in output; `unexpected keyword argument 'similarity_threshold'` | `e063a01` | `a4df664` |
| 4 | `max_per_source` optionally limits chunks per document and defaults to no cap. | `test_optional_per_source_cap_increases_context_diversity` | `unexpected keyword argument 'max_per_source'` | `5f2e0b7` | `846a615` |
| 5 | All chunk metadata is preserved unchanged and equal scores retain original-index order. | `test_metadata_is_unchanged_and_score_ties_keep_original_order` | returned chunks omitted `document`, `page`, `label`, `kind`, and `chunk_id` | `9e4e96e` | `1bbc0a8` |
| 6 | Existing retrieval paths use an optional wrapper that defaults off; a fake retriever proves enabled packing. | `test_retrieval_integration_is_default_off_and_can_pack_fake_results` | `ImportError: cannot import name 'retrieve_context'` | `0a633e8` | `3d3163b` |

## Refactor and verification

- `95752bc` — reused normalized text fingerprints to avoid repeated normalization/tokenization without changing behavior.
- `d87b627` — added the repository's first GitHub Actions pytest workflow.
- Final local command: `.venv/bin/python -m pytest -q`.
- Final result: `68 passed, 1 skipped`; the skipped case is the explicitly opted-in paid API integration test.
