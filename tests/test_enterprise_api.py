import time
import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.enterprise_api import AskResult, BackendUnavailable, create_app


class StubService:
    mode = "mock"

    def __init__(self, result=None, error=None, delay=0):
        self.result = result
        self.error = error
        self.delay = delay

    def ask(self, _question):
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def test_normal_query_returns_structured_answer_and_sources():
    result = AskResult("BetaDet scores 33.0 [S1].", [{"citation": "S1", "document": "beta.pdf", "page": 5}])
    response = TestClient(create_app(StubService(result))).post("/api/ask", json={"question": "Which is better?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "BetaDet scores 33.0 [S1].",
        "sources": [{"citation": "S1", "document": "beta.pdf", "page": 5}],
        "status": "ok",
        "mode": "mock",
    }


def test_empty_question_is_rejected_before_backend_call():
    response = TestClient(create_app(StubService())).post("/api/ask", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["status"] == "invalid_request"
    assert response.json()["sources"] == []


def test_no_related_material_returns_no_evidence():
    result = AskResult("Insufficient evidence.", [], status="no_evidence")
    response = TestClient(create_app(StubService(result))).post("/api/ask", json={"question": "Unknown fact"})

    assert response.status_code == 200
    assert response.json()["status"] == "no_evidence"
    assert response.json()["sources"] == []


def test_backend_unavailable_returns_503():
    service = StubService(error=BackendUnavailable("rag_backend_unavailable"))
    response = TestClient(create_app(service)).post("/api/ask", json={"question": "Question"})

    assert response.status_code == 503
    assert response.json()["status"] == "backend_unavailable"
    assert response.json()["message"] == "rag_backend_unavailable"


def test_request_timeout_returns_504():
    service = StubService(result=AskResult("late", []), delay=0.05)
    response = TestClient(create_app(service, request_timeout=0.01)).post(
        "/api/ask", json={"question": "Slow question"})

    assert response.status_code == 504
    assert response.json()["status"] == "timeout"


def test_n8n_workflow_is_inactive_and_uses_docker_host_gateway():
    workflow = json.loads((Path(__file__).parents[1] / "n8n" /
                           "enterprise_ai_assistant.workflow.json").read_text(encoding="utf-8"))
    nodes = {node["name"]: node for node in workflow["nodes"]}

    assert workflow["active"] is False
    assert nodes["Webhook"]["typeVersion"] == 2.2
    assert nodes["Call FastAPI Hybrid RAG"]["typeVersion"] == 4.5
    assert nodes["Call FastAPI Hybrid RAG"]["parameters"]["url"] == (
        "http://host.docker.internal:8000/api/ask")
    assert nodes["Return Answer"]["typeVersion"] == 1.5
