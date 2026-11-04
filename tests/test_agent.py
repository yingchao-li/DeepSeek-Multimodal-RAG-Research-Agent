import unittest
from unittest.mock import patch
from src.agentic_rag import AgentLimits, run_agentic_rag
from src.embeddings import query_embedder
from src.vector_store import build_retrieval_index
from tests.helpers import FakeClient
from tests.test_retrieval import ToyEmbedding


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{'source': 'a.pdf', 'page': 2, 'text': 'apple network architecture'},
                     {'source': 'b.pdf', 'page': 3, 'text': 'banana network architecture'}]
        self.model = ToyEmbedding()
        self.index = build_retrieval_index(self.docs, self.model)

    def run_agent(self, question, client, **kwargs):
        return run_agentic_rag(question, [], index=self.index, chunks=self.docs, top_k=1,
                               embed_fn=query_embedder(self.model), client=client, allow_web=False,
                               lang_code='en', **kwargs)

    def test_original_route_grade_generate_verify(self):
        client = FakeClient(['{"route":"kb"}', '{"results":[{"idx":0,"relevant":true}]}',
                             'Supported [S1].', '{"pass":true,"final_answer":"Supported [S1]."}'])
        result = self.run_agent('apple network', client,
                                limits=AgentLimits(max_iterations=1, max_api_calls=4))
        self.assertEqual(result['answer'], 'Supported [S1].')
        self.assertEqual(result['evidence'][0]['page'], 2)
        self.assertEqual(result['trace']['api_calls'], 4)

    def test_cross_document_source_constrained_retrieval(self):
        client = FakeClient(['{"route":"kb"}', 'Comparison [S1] [S2].',
                             '{"pass":true,"final_answer":"Comparison [S1] [S2]."}'])
        result = self.run_agent('Compare the network architectures', client,
                                limits=AgentLimits(max_iterations=1, max_api_calls=3))
        self.assertEqual({d['source'] for d in result['retrieved_docs']}, {'a.pdf', 'b.pdf'})
        self.assertEqual(len(result['trace']['steps'][0]['searches']), 2)

    def test_iterative_evidence_accumulation(self):
        client = FakeClient(['Answer [S1] [S2]', '{"pass":true,"final_answer":"Answer [S1] [S2]"}'])
        with patch('src.agentic_rag.route_query', return_value={'route':'kb'}), \
             patch('src.agentic_rag.grade_documents', side_effect=lambda q, docs, c: (docs, {})), \
             patch('src.agentic_rag.assess_evidence', return_value=(False, 'banana', {'sufficient':False})):
            result = self.run_agent('apple', client, limits=AgentLimits(max_iterations=2))
        self.assertEqual(len(result['trace']['steps']), 2)
        self.assertEqual(len(result['retrieved_docs']), 2)
        self.assertEqual(client.calls[0]['max_tokens'], 1024)

    def test_zero_tool_budget_and_minimal_api_budget(self):
        client = FakeClient()
        result = self.run_agent('apple', client, limits=AgentLimits(max_tool_calls=0, max_api_calls=2))
        self.assertIn('insufficient', result['answer'])
        self.assertEqual(result['trace']['api_calls'], 0)
        self.assertEqual(result['trace']['tool_calls'], 0)

    def test_hallucinated_citation_removed_after_verification(self):
        client = FakeClient(['draft [S99]', '{"pass":true,"final_answer":"draft [S99]"}'])
        result = self.run_agent('apple', client, limits=AgentLimits(max_api_calls=2, max_iterations=1))
        self.assertNotIn('[S99]', result['answer'])
        self.assertEqual(result['answer'], 'The available evidence is insufficient for a reliable answer. Add relevant documents, identify a page or figure, or enable web search.')
        self.assertEqual(result['evidence'], [])
        self.assertEqual(result['trace']['api_calls'], 2)
