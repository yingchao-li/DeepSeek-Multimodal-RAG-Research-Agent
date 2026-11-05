"""Thin evaluation adapters over the project's real Plain and Agentic RAG paths."""
from __future__ import annotations
from dataclasses import dataclass
import time

from src.agentic_rag import AgentLimits, run_agentic_rag
from src.citations import evidence_records
from src.llm import generate_rag_answer
from src.guardrails import is_abstention
from src.context_packer import retrieve_context
from src.vector_store import retrieve
from eval.cost import usage_totals


@dataclass
class EvalContext:
    retrieval_index: object
    chunks: list[dict]
    embed_fn: object
    retrieval_config: object
    client: object
    agent_limits: AgentLimits
    lang_code: str = "en"


def _usage(client, sequence):
    records = [dict(r) for r in client.records if r.get("sequence", 0) > sequence]
    return records, usage_totals(records)


def _evidence(docs):
    fields = ("chunk_id", "source", "page", "section", "label", "kind", "text")
    return [{key: doc[key] for key in fields if doc.get(key) is not None} for doc in docs]


def answer_baseline(question, index, k):
    """Question -> shared retrieval -> context -> existing DeepSeek RAG generation."""
    ctx: EvalContext = index
    started, sequence = time.perf_counter(), ctx.client.total_calls
    retrieval_started = time.perf_counter()
    docs = retrieve_context(retrieve, question, ctx.retrieval_index, ctx.chunks, k, ctx.embed_fn,
                            config=ctx.retrieval_config)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    try:
        answer = generate_rag_answer(question, docs, [], ctx.client, lang_code=ctx.lang_code)
        if is_abstention(answer):
            docs = []
        status, error = "ok", None
    except Exception as exc:
        answer, status, error = "", "error", str(exc)
    records, usage = _usage(ctx.client, sequence)
    return {"pipeline": "baseline", "status": status, "answer": answer,
            "evidence": _evidence(docs), "evidence_records": evidence_records(docs),
            "usage": usage, "api_records": records,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "retrieval_latency_ms": round(retrieval_ms, 2), "agent_trace": None,
            "error": error, "retry_count": 0}


def answer_agentic(question, index, k):
    """Run the existing bounded Agentic RAG implementation without simplifying it."""
    ctx: EvalContext = index
    started, sequence = time.perf_counter(), ctx.client.total_calls
    try:
        outcome = run_agentic_rag(
            question, [], index=ctx.retrieval_index, chunks=ctx.chunks, top_k=k,
            embed_fn=ctx.embed_fn, client=ctx.client, router_domain="library",
            lang_code=ctx.lang_code, retrieval_config=ctx.retrieval_config,
            limits=ctx.agent_limits, allow_web=False, decompose=True,
        )
        docs, trace = outcome["retrieved_docs"], outcome["trace"]
        answer = outcome["answer"]
        # A retained evidence list does not make an interrupted generation valid.
        status = "error" if trace.get("warnings") else "ok"
        error = ",".join(trace.get("warnings", [])) or None
    except Exception as exc:
        docs, trace, answer, status, error = [], {}, "", "error", str(exc)
    records, usage = _usage(ctx.client, sequence)
    retrieval_ms = sum(float(search.get("retrieval", {}).get("latency_ms", 0) or 0)
                       for step in trace.get("steps", []) for search in step.get("searches", []))
    return {"pipeline": "agentic", "status": status, "answer": answer,
            "evidence": _evidence(docs), "evidence_records": evidence_records(docs),
            "usage": usage, "api_records": records,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "retrieval_latency_ms": round(retrieval_ms, 2), "agent_trace": trace,
            "error": error, "retry_count": 0}
