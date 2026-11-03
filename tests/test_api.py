import unittest
from src.api import APIError, BudgetExceeded, ManagedClient, api_budget, request
from tests.helpers import FakeClient


class APITests(unittest.TestCase):
    def test_budget_applies_to_mock_clients(self):
        fake = FakeClient()
        with api_budget(max_calls=1, max_output_tokens=64):
            request(fake, model="text-test", messages=[])
            with self.assertRaises(BudgetExceeded):
                request(fake, model="text-test", messages=[])
        self.assertEqual(fake.calls[0]["max_tokens"], 64)

    def test_usage_is_recorded_without_prompt_or_key(self):
        managed = ManagedClient(FakeClient())
        managed.create(model="test", messages=[{"role": "user", "content": "private"}])
        self.assertEqual(managed.records[0]["prompt_tokens"], 12)
        self.assertNotIn("private", str(managed.records))
        self.assertIsNone(managed.records[0]["estimated_cost_usd"])

    def test_errors_never_expose_server_body(self):
        fake = FakeClient()
        def fail(**kwargs):
            raise RuntimeError("secret server body")
        fake.create = fail
        managed = ManagedClient(fake)
        with self.assertRaisesRegex(APIError, "connection_or_server") as result:
            managed.create(model="test", messages=[])
        self.assertNotIn("secret", str(result.exception))
