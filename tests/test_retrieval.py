import unittest
from unittest.mock import patch
import numpy as np
from src.vector_store import (BM25, IndexMismatch, RetrievalConfig, build_retrieval_index,
                              reciprocal_rank_fusion, retrieve)
from src.embeddings import query_embedder


class ToyEmbedding:
    fingerprint = 'toy-v1'
    def encode(self, texts, **kwargs):
        return np.asarray([[1, 0] if 'apple' in text else [0, 1] for text in texts], dtype='float32')


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{'source': 'a.txt', 'text': 'apple red fruit'},
                     {'source': 'b.txt', 'text': 'banana yellow fruit'},
                     {'source': 'c.txt', 'text': 'banana detection experiment'}]
        self.model = ToyEmbedding()
        self.index = build_retrieval_index(self.docs, self.model)
        self.encode = query_embedder(self.model)

    def test_dense_and_hybrid(self):
        for strategy in ('dense', 'hybrid'):
            hits = retrieve('apple', self.index, self.docs, 2, self.encode,
                            config=RetrievalConfig(strategy=strategy))
            self.assertEqual(hits[0]['source'], 'a.txt')
            self.assertTrue(hits[0]['chunk_id'])

    def test_rrf_dedup_and_lexical_search(self):
        self.assertEqual(reciprocal_rank_fusion([[(1, 0), (0, 0)], [(0, 0)]] )[0][0], 0)
        self.assertEqual(BM25(['红色苹果好吃', '香蕉']).search('苹果', 1)[0][0], 0)

    def test_fingerprint_and_corpus_guard(self):
        self.encode.fingerprint = 'different-same-dimension'
        with self.assertRaises(IndexMismatch):
            retrieve('apple', self.index, self.docs, 1, self.encode)
        self.encode.fingerprint = 'toy-v1'
        self.docs[0]['text'] = 'changed'
        with self.assertRaises(IndexMismatch):
            retrieve('apple', self.index, self.docs, 1, self.encode)

    def test_real_reranker_contract_with_mock_scores(self):
        class FakeReranker:
            def score(self, query, docs):
                return [100 if 'experiment' in d else 0 for d in docs]
        hits = retrieve('apple', self.index, self.docs, 2, self.encode,
                        config=RetrievalConfig(strategy='hybrid_rerank'), reranker=FakeReranker())
        self.assertEqual(hits[0]['source'], 'c.txt')
        self.assertTrue(hits[0]['retrieval']['rerank_applied'])

    def test_missing_reranker_does_not_break_retrieval(self):
        with patch('src.reranking.load_reranker', side_effect=ImportError):
            hits = retrieve('apple', self.index, self.docs, 2, self.encode,
                            config=RetrievalConfig(strategy='hybrid_rerank'))
        self.assertFalse(hits[0]['retrieval']['rerank_applied'])
        self.assertEqual(hits[0]['retrieval']['warning'], 'reranker_unavailable')
