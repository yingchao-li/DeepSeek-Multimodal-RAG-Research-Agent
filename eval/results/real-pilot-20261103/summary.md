# Real DeepSeek API Pilot Evaluation — Synthetic Dataset

> This is an unreviewed synthetic pilot, not a formal benchmark or CV claim.

Questions: 8; K=4; seed=42.

| Pipeline | Correctness | Faithfulness | Abstention accuracy | Hit@K | Unsupported claims | Mean / Median / P95 latency | API calls | Prompt / Completion tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.938 | 1.000 | 0.875 | 0.833 | 0.000 | 12382 / 4478 / 69846 ms | 8 | 4084 / 1468 |
| agentic | 0.750 | 0.875 | 0.750 | 0.833 | 0.000 | 38180 / 25092 / 105012 ms | 40 | 13374 / 4188 |

Generator and Judge usage are stored separately in `summary.csv`. Cost is N/A unless reliable prices are configured.

Every candidate and pilot item is marked `needs_human_review=true`. LLM-as-Judge may favor outputs from a related model family and requires human review.
