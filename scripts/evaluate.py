#!/usr/bin/env python3
"""Run from the app root. No paid APIs without both explicit flags."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.embeddings import SentenceEmbeddingAdapter, LIGHTWEIGHT_MODEL
from src.evaluation import DEFAULT_DATASET, run_benchmark, save_report
from src.vector_store import RetrievalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--output', type=Path, default=Path('reports'))
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--rerank-k', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--embedding-model', default=LIGHTWEIGHT_MODEL)
    parser.add_argument('--reranker-model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--evaluate-answers', action='store_true')
    parser.add_argument('--allow-paid', action='store_true')
    args = parser.parse_args()
    if args.evaluate_answers and not args.allow_paid:
        parser.error('--evaluate-answers requires --allow-paid')
    config = RetrievalConfig(candidate_k=args.candidate_k, rerank_k=args.rerank_k,
                             reranker_model=args.reranker_model, allow_download=args.allow_download)
    report = run_benchmark(args.dataset, k=args.k, config=config,
                           embed_model=SentenceEmbeddingAdapter(args.embedding_model, allow_download=args.allow_download),
                           evaluate_answers=args.evaluate_answers, allow_paid=args.allow_paid,
                           repeats=args.repeats,
                           progress=lambda row:print(json.dumps(row),flush=True))
    save_report(report, args.output)
    print(json.dumps({'summary': report['summary'], 'api_calls': report['api_calls'],
                      'output': str(args.output.resolve())}, indent=2))


if __name__ == '__main__':
    main()
