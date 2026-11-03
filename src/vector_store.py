"""Original FAISS baseline plus versioned dense/BM25/RRF retrieval."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
import time
from typing import Any
import faiss
import numpy as np


def chunk_id(doc):
    fields = {key: doc.get(key) for key in ('source', 'page', 'kind', 'label', 'section', 'text')}
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:20]


def tokenize(text):
    text = text.casefold()
    words = re.findall(r'[\w]+', text, flags=re.UNICODE)
    # Character bigrams give Chinese text lexical recall without a large tokenizer.
    chinese = re.findall(r'[\u3400-\u9fff]+', text)
    return words + [run[i:i+2] for run in chinese for i in range(len(run)-1)]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.freqs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(f.values()) for f in self.freqs]
        self.avg_len = sum(self.lengths) / max(1, len(texts)) or 1
        df = Counter(term for f in self.freqs for term in f)
        self.idf = {term: math.log(1 + (len(texts)-count+0.5)/(count+0.5)) for term, count in df.items()}

    def search(self, query, k):
        terms = set(tokenize(query))
        scores = []
        for i, freq in enumerate(self.freqs):
            norm = self.k1 * (1-self.b+self.b*self.lengths[i]/self.avg_len)
            score = sum(self.idf.get(t, 0)*freq[t]*(self.k1+1)/(freq[t]+norm)
                        for t in terms if freq[t])
            if score > 0:
                scores.append((i, score))
        return sorted(scores, key=lambda x: (-x[1], x[0]))[:k]


@dataclass
class RetrievalConfig:
    strategy: str = 'dense'
    candidate_k: int = 20
    rerank_k: int = 12
    rrf_k: int = 60
    reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    allow_download: bool = False
    device: str = 'cpu'

    def __post_init__(self):
        if self.strategy not in ('dense', 'hybrid', 'hybrid_rerank'):
            raise ValueError('Unknown retrieval strategy')
        if min(self.candidate_k, self.rerank_k, self.rrf_k) < 1:
            raise ValueError('Retrieval parameters must be positive')


@dataclass
class RetrievalIndex:
    dense: Any
    sparse: BM25
    embedding_fingerprint: str
    corpus_fingerprint: str
    dimension: int
    version: int = 1


class IndexMismatch(ValueError):
    pass


def corpus_fingerprint(chunks):
    return hashlib.sha256('|'.join(chunk_id(c) for c in chunks).encode()).hexdigest()


def build_faiss_index(vectors):
    vectors = np.ascontiguousarray(vectors, dtype='float32')
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be a non-empty finite matrix')
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_retrieval_index(chunks, embed_model):
    from src.embeddings import embed_texts
    if not chunks:
        raise ValueError('Cannot index empty documents')
    for chunk in chunks:
        chunk.setdefault('chunk_id', chunk_id(chunk))
    vectors = embed_texts([c['text'] for c in chunks], embed_model)
    return RetrievalIndex(build_faiss_index(vectors), BM25([c['text'] for c in chunks]),
                          getattr(embed_model, 'fingerprint', 'legacy'),
                          corpus_fingerprint(chunks), vectors.shape[1])


def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, item in enumerate(ranking, 1):
            idx = item[0]
            if idx not in seen:
                scores[idx] = scores.get(idx, 0) + 1/(rrf_k+rank)
                seen.add(idx)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def retrieve(query: str, index, chunks: list[dict], k: int, embed_fn, *, config=None, reranker=None,
             source_filter=None):
    if not chunks or k <= 0:
        return []
    config = config or RetrievalConfig()
    start = time.perf_counter()
    if isinstance(index, RetrievalIndex):
        fingerprint = getattr(embed_fn, 'fingerprint', None)
        if fingerprint and fingerprint != index.embedding_fingerprint:
            raise IndexMismatch('embedding_changed_rebuild_index')
        if index.version != 1 or corpus_fingerprint(chunks) != index.corpus_fingerprint:
            raise IndexMismatch('corpus_changed_rebuild_index')
        dense, sparse = index.dense, index.sparse
    else:
        if getattr(embed_fn, 'fingerprint', None):
            raise IndexMismatch('unversioned_index_rebuild_required')
        dense, sparse = index, BM25([c['text'] for c in chunks])
    query_vec = np.ascontiguousarray(embed_fn([query]), dtype='float32')
    if query_vec.ndim != 2 or query_vec.shape[1] != dense.d or not np.isfinite(query_vec).all():
        raise IndexMismatch('embedding_dimension_rebuild_index')
    candidates = min(len(chunks), max(k, config.candidate_k) if config.strategy != 'dense' else k)
    if source_filter:
        candidates = len(chunks)
    scores, indices = dense.search(query_vec, candidates)
    dense_hits = [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if 0 <= i < len(chunks)]
    if source_filter:
        dense_hits = [(i, s) for i, s in dense_hits if chunks[i].get('source') == source_filter]
    fused = dense_hits
    if config.strategy != 'dense':
        sparse_hits = sparse.search(query, candidates)
        if source_filter:
            sparse_hits = [(i, s) for i, s in sparse_hits if chunks[i].get('source') == source_filter]
        fused = reciprocal_rank_fusion([dense_hits, sparse_hits], config.rrf_k)
    docs = [{**chunks[i], 'chunk_id': chunks[i].get('chunk_id') or chunk_id(chunks[i]),
             'score': float(score)} for i, score in fused]
    applied, warning = False, None
    if config.strategy == 'hybrid_rerank' and docs:
        try:
            if reranker is None:
                from src.reranking import load_reranker
                reranker = load_reranker(config.reranker_model, config.device, config.allow_download)
            selected = docs[:config.rerank_k]
            rerank_scores = reranker.score(query, [d['text'] for d in selected])
            if len(rerank_scores) != len(selected) or not np.isfinite(rerank_scores).all():
                raise ValueError('Invalid reranker scores')
            for doc, score in zip(selected, rerank_scores):
                doc['rerank_score'] = float(score)
            docs = sorted(selected, key=lambda d: -d['rerank_score']) + docs[len(selected):]
            applied = True
        except Exception:
            warning = 'reranker_unavailable'
    for doc in docs:
        doc['retrieval'] = {'strategy': config.strategy, 'rerank_applied': applied,
                            'warning': warning, 'latency_ms': round((time.perf_counter()-start)*1000, 2)}
    return docs[:k]
