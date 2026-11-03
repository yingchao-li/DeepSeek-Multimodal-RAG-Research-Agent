import math
import tempfile
import unittest
from unittest.mock import patch
from src.evaluation import judge_answer, retrieval_metrics, run_benchmark, save_report
from tests.helpers import FakeClient
from tests.test_retrieval import ToyEmbedding


class EvaluationTests(unittest.TestCase):
    def test_known_metrics(self):
        metrics = retrieval_metrics(['a', 'c'], ['a', 'b'], 2)
        self.assertEqual(metrics['recall_at_k'], 0.5)
        self.assertEqual(metrics['precision_at_k'], 0.5)
        self.assertEqual(metrics['mrr'], 1)
        self.assertAlmostEqual(metrics['ndcg_at_k'], 1/(1+1/math.log2(3)))
        self.assertEqual(retrieval_metrics([], [], 3)['mrr'], 0)
        duplicate = retrieval_metrics(['a', 'a', 'b'], ['a', 'b'], 2)
        self.assertEqual(duplicate['recall_at_k'], 0.5)
        self.assertEqual(duplicate['precision_at_k'], 0.5)
        self.assertEqual(retrieval_metrics(['x','y','a'],['a'],2)['mrr'],1/3)

    def test_reports_are_consistent_and_unavailable_reranker_is_not_measured(self):
        with patch('src.reranking.load_reranker', side_effect=ImportError):
            report = run_benchmark(embed_model=ToyEmbedding())
        self.assertEqual(report['api_calls'], 0)
        self.assertEqual(len(report['queries']), 16)
        self.assertEqual(report['summary'][2]['status'], 'not_measured_model_unavailable')
        self.assertNotIn('recall_at_k', report['summary'][2])
        with tempfile.TemporaryDirectory() as folder:
            path = save_report(report, folder)
            self.assertTrue((path/'queries.csv').is_file())
            self.assertTrue((path/'retrieval.json').is_file())

    def test_paid_evaluation_requires_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'explicit'):
            run_benchmark(embed_model=ToyEmbedding(), evaluate_answers=True)

    def test_mock_answer_quality_judge(self):
        client = FakeClient(['{"faithfulness":1,"answer_relevancy":0.8,"context_precision":1,"context_recall":0.5}'])
        result = judge_answer(client, 'question', 'answer', [{'text':'context'}], 'reference')
        self.assertEqual(result['faithfulness'], 1)
        self.assertEqual(result['context_recall'], 0.5)
