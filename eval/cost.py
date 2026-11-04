"""Token and optional configured-cost aggregation."""
from __future__ import annotations


def usage_totals(records):
    return {
        "api_calls": len(records),
        "prompt_tokens": sum(int(r.get("prompt_tokens", 0) or 0) for r in records),
        "completion_tokens": sum(int(r.get("completion_tokens", 0) or 0) for r in records),
        "total_tokens": sum(int(r.get("total_tokens", 0) or 0) for r in records),
    }


def estimated_cost(records, prices):
    """Return USD or None when any used model has no reliable configured price."""
    total = 0.0
    for record in records:
        model = record.get("response_model") or record.get("model")
        rate = (prices or {}).get(model)
        if not rate:
            return None
        total += (record.get("prompt_tokens", 0) * float(rate["input"]) +
                  record.get("completion_tokens", 0) * float(rate["output"])) / 1_000_000
    return total

