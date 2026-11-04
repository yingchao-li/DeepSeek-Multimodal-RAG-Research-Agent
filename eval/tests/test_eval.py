import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from eval.cost import estimated_cost, usage_totals
from eval.judge import parse_judge_json
from eval.metrics import latency_stats, retrieval_hit_at_k, summarize
from eval.run import main


class EvaluationTests(unittest.TestCase):
    def test_hit_at_k(self):
        docs = [{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 2}]
        self.assertEqual(retrieval_hit_at_k(docs, [{"source": "b.pdf", "page": 2}], 2), 1.0)
        self.assertEqual(retrieval_hit_at_k(docs, [{"source": "b.pdf", "page": 2}], 1), 0.0)

    def test_judge_json_parser(self):
        raw = '```json\n{"correctness":"Correct","faithfulness":1,"abstained":false,' \
              '"abstention_correct":true,"unsupported_claim_rate":0,"reason":"supported"}\n```'
        self.assertEqual(parse_judge_json(raw)["correctness"], "Correct")

    def test_abstention_summary(self):
        rows = [{"pipeline":"baseline","status":"ok","latency_ms":10,"usage":{},
                 "evidence":[],"dataset":{"unanswerable":True,"gold_evidence":[]},
                 "judge":{"status":"ok","correctness":"Correct","faithfulness":1,
                          "abstention_correct":True,"unsupported_claim_rate":0}}]
        self.assertEqual(summarize(rows, "baseline", 4)["abstention_accuracy"], 1.0)

    def test_latency_statistics(self):
        self.assertEqual(latency_stats([10, 20, 30, 40])["median_ms"], 25)
        self.assertEqual(latency_stats([10, 20, 30, 40])["p95_ms"], 40)

    def test_cost_and_usage(self):
        records = [{"response_model":"m","prompt_tokens":100,"completion_tokens":50,"total_tokens":150}]
        self.assertEqual(usage_totals(records)["total_tokens"], 150)
        self.assertAlmostEqual(estimated_cost(records, {"m":{"input":1,"output":2}}), .0002)
        self.assertIsNone(estimated_cost(records, {}))

    def test_dry_run_runner(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--dry-run", "--mode", "both", "--dataset", "eval/dataset.jsonl"]), 0)
        self.assertEqual(json.loads(output.getvalue())["questions"], 8)


if __name__ == "__main__":
    unittest.main()
