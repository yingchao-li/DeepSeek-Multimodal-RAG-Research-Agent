"""Local FastAPI adapter for n8n -> Hybrid RAG -> structured answer workflows."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api import ManagedClient, setting
from src.context_packer import retrieve_context
from src.embeddings import SentenceEmbeddingAdapter, query_embedder
from src.guardrails import is_abstention
from src.llm import generate_rag_answer
from src.paths import APP_ROOT
from src.vector_store import RetrievalConfig, build_retrieval_index, retrieve, tokenize


class BackendUnavailable(RuntimeError):
    """The configured local retrieval or generation backend cannot start."""


@dataclass
class AskResult:
    answer: str
    sources: list[dict]
    status: str = "ok"
    mode: str = "mock"


class AskRequest(BaseModel):
    question: str = ""


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "what", "which", "who",
    "why", "with",
}


def _boolean_setting(name, default=False):
    value = str(setting(name, "true" if default else "false")).strip().casefold()
    return value in {"1", "true", "yes", "on"}


def _number_setting(name, default, minimum, maximum, cast=float):
    try:
        value = cast(setting(name, default))
    except (TypeError, ValueError):
        return default
    return value if minimum <= value <= maximum else default


def _source_records(docs):
    records = []
    for index, doc in enumerate(docs, 1):
        record = {
            "citation": f"S{index}",
            "document": doc.get("source", "unknown"),
            "page": doc.get("page"),
            "figure_table": doc.get("label"),
            "chunk_id": doc.get("chunk_id"),
            "score": doc.get("score"),
        }
        records.append({key: value for key, value in record.items() if value is not None})
    return records


class EnterpriseRAGService:
    """Lazy adapter around the repository's existing hybrid retriever and LLM wrapper."""

    def __init__(self, *, mode=None, corpus_path=None):
        self.mode = (mode or setting("ENTERPRISE_ASSISTANT_MODE", "mock")).strip().casefold()
        configured_path = Path(corpus_path or setting(
            "RAG_CORPUS_PATH", str(APP_ROOT / "evaluation" / "example.json")))
        self.corpus_path = configured_path if configured_path.is_absolute() else APP_ROOT / configured_path
        self._lock = threading.Lock()
        self._loaded = False
        self._chunks = None
        self._index = None
        self._embed_fn = None
        self._client = None
        self._retrieval = RetrievalConfig(
            strategy="hybrid",
            candidate_k=_number_setting("RAG_CANDIDATE_K", 20, 1, 100, int),
            rerank_k=12,
        )
        self._top_k = _number_setting("RAG_TOP_K", 5, 1, 20, int)
        self._mock_delay = _number_setting("ENTERPRISE_MOCK_DELAY_SECONDS", 0.0, 0.0, 60.0)

    def _load(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if self.mode not in {"mock", "real"}:
                raise BackendUnavailable("invalid_assistant_mode")
            if not self.corpus_path.is_file():
                raise BackendUnavailable("rag_corpus_missing")
            try:
                payload = json.loads(self.corpus_path.read_text(encoding="utf-8"))
                chunks = payload.get("chunks") if isinstance(payload, dict) else payload
                if not isinstance(chunks, list) or not chunks:
                    raise ValueError
                chunks = [dict(chunk) for chunk in chunks if isinstance(chunk, dict)
                          and str(chunk.get("text", "")).strip()]
                if not chunks:
                    raise ValueError
                model = SentenceEmbeddingAdapter(
                    str(setting("EMBEDDING_MODEL", "all-MiniLM-L6-v2")),
                    device=str(setting("MODEL_DEVICE", "cpu")),
                    allow_download=_boolean_setting("ALLOW_MODEL_DOWNLOAD", False),
                )
                self._index = build_retrieval_index(chunks, model)
                self._embed_fn = query_embedder(model)
                self._chunks = chunks
                self._client = ManagedClient()
            except BackendUnavailable:
                raise
            except Exception:
                raise BackendUnavailable("rag_backend_unavailable") from None
            self._loaded = True

    def _has_lexical_evidence(self, question):
        terms = {term for term in tokenize(question) if len(term) > 2 and term not in _STOP_WORDS}
        return bool(terms and any(terms & set(tokenize(chunk.get("text", "")))
                                  for chunk in self._chunks))

    def ask(self, question):
        self._load()
        if self._mock_delay:
            time.sleep(self._mock_delay)
        if not self._has_lexical_evidence(question):
            return AskResult(
                answer="Insufficient evidence in the configured document corpus.",
                sources=[], status="no_evidence", mode=self.mode,
            )

        docs = retrieve_context(
            retrieve, question, self._index, self._chunks, self._top_k, self._embed_fn,
            config=self._retrieval,
            enabled=_boolean_setting("RAG_CONTEXT_PACKING", False),
            max_tokens=_number_setting("RAG_CONTEXT_MAX_TOKENS", 4096, 1, 100000, int),
            max_per_source=(_number_setting("RAG_MAX_PER_SOURCE", 3, 1, 20, int)
                            if setting("RAG_MAX_PER_SOURCE") is not None else None),
        )
        if not docs:
            return AskResult(
                answer="Insufficient evidence in the configured document corpus.",
                sources=[], status="no_evidence", mode=self.mode,
            )

        if self.mode == "mock":
            excerpts = [f"[S{index}] {doc.get('text', '').strip()}"
                        for index, doc in enumerate(docs[:3], 1)]
            answer = "[MOCK GENERATION — RETRIEVED EVIDENCE]\n" + "\n".join(excerpts)
        else:
            if not setting("DEEPSEEK_API_KEY"):
                raise BackendUnavailable("missing_api_key")
            answer = generate_rag_answer(question, docs, [], self._client, lang_code="en")
            if is_abstention(answer):
                return AskResult(answer=answer, sources=[], status="no_evidence", mode=self.mode)
        return AskResult(answer=answer, sources=_source_records(docs), mode=self.mode)


def _payload(result):
    return {
        "answer": result.answer,
        "sources": result.sources,
        "status": result.status,
        "mode": result.mode,
    }


def _error(status, message, mode):
    return {"answer": "", "sources": [], "status": status, "mode": mode, "message": message}


def create_app(service=None, request_timeout=None):
    service = service or EnterpriseRAGService()
    timeout = request_timeout if request_timeout is not None else _number_setting(
        "RAG_REQUEST_TIMEOUT_SECONDS", 30.0, 0.01, 300.0)
    app = FastAPI(title="Enterprise AI Assistant API", version="1.0.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "mode": service.mode}

    @app.post("/api/ask")
    async def ask(request: AskRequest):
        question = request.question.strip()
        if not question:
            return JSONResponse(status_code=400,
                                content=_error("invalid_request", "question must not be empty", service.mode))
        try:
            result = await asyncio.wait_for(asyncio.to_thread(service.ask, question), timeout=timeout)
            return _payload(result)
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504,
                                content=_error("timeout", "RAG backend request timed out", service.mode))
        except BackendUnavailable as exc:
            return JSONResponse(status_code=503,
                                content=_error("backend_unavailable", str(exc), service.mode))
        except Exception:
            return JSONResponse(status_code=502,
                                content=_error("backend_error", "RAG backend failed", service.mode))

    return app


app = create_app()
