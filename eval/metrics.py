"""Deterministic pilot metrics."""
from __future__ import annotations
import math
import statistics


def evidence_matches(doc, gold):
    if gold.get("chunk_id") and doc.get("chunk_id") == gold["chunk_id"]:
        return True
    if gold.get("source") and doc.get("source") != gold["source"]:
        return False
    if gold.get("page") is not None and doc.get("page") != gold["page"]:
        return False
    if gold.get("label") and str(doc.get("label", "")).casefold() != str(gold["label"]).casefold():
        return False
    return bool(gold.get("source"))


def retrieval_hit_at_k(evidence, gold_evidence, k):
    if not gold_evidence:
        return None
    return float(any(evidence_matches(doc, gold) for doc in evidence[:k] for gold in gold_evidence))


def latency_stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"mean_ms": None, "median_ms": None, "p95_ms": None}
    rank = max(0, math.ceil(0.95 * len(values)) - 1)
    return {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
            "p95_ms": values[rank]}


def summarize(rows, pipeline, k):
    selected = [r for r in rows if r.get("pipeline") == pipeline and r.get("status") == "ok"]
    judges = [r["judge"] for r in selected if r.get("judge", {}).get("status") == "ok"]
    answerable = [r for r in selected if not r.get("dataset", {}).get("unanswerable")]
    hits = [retrieval_hit_at_k(r.get("evidence", []), r["dataset"].get("gold_evidence", []), k)
            for r in answerable]
    correctness = {"Correct": 1.0, "Partial": 0.5, "Incorrect": 0.0}
    usage = {key: sum(r.get("usage", {}).get(key, 0) for r in selected)
             for key in ("api_calls", "prompt_tokens", "completion_tokens", "total_tokens")}
    latency = latency_stats([r["latency_ms"] for r in selected])
    return {
        "pipeline": pipeline, "questions": len(selected),
        "retrieval_hit_at_k": sum(hits) / len(hits) if hits else None,
        "answer_correctness": (sum(correctness.get(j.get("correctness"), 0) for j in judges) / len(judges)
                               if judges else None),
        "faithfulness": sum(float(j.get("faithfulness", 0)) for j in judges) / len(judges) if judges else None,
        "abstention_accuracy": (sum(bool(j.get("abstention_correct")) for j in judges) / len(judges)
                                if judges else None),
        "unsupported_claim_rate": (sum(float(j.get("unsupported_claim_rate", 0)) for j in judges) / len(judges)
                                   if judges else None),
        **latency, **usage,
    }

