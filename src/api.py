"""OpenAI-compatible client with bounded calls, safe errors and usage accounting.

No key is needed to import modules or start the UI. The SDK is constructed only
on the first real request. Test clients can be injected without network access.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import os
import time
from types import SimpleNamespace


class APIError(RuntimeError):
    """Public error: contains a category, never a server body or credentials."""


class BudgetExceeded(APIError):
    pass


def setting(name, default=None):
    value = os.environ.get(name)
    if value is not None:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def text_model():
    return str(setting("DEEPSEEK_TEXT_MODEL", "deepseek-chat"))


def vision_model():
    return str(setting("DEEPSEEK_VISION_MODEL", "deepseek-flash"))


@dataclass
class CallBudget:
    max_calls: int = 8
    calls: int = 0
    max_output_tokens: int = 1024
    max_input_chars: int = 100000

    def consume(self):
        if self.calls >= self.max_calls:
            raise BudgetExceeded("api_call_limit")
        self.calls += 1


_BUDGET = ContextVar("deepseek_budget", default=None)


@contextmanager
def api_budget(max_calls=8, max_output_tokens=1024, max_input_chars=100000):
    budget = CallBudget(max_calls=max_calls, max_output_tokens=max_output_tokens,
                        max_input_chars=max_input_chars)
    token = _BUDGET.set(budget)
    try:
        yield budget
    finally:
        _BUDGET.reset(token)


def request(client, **kwargs):
    """All application LLM calls pass here, including injected mock clients."""
    budget = _BUDGET.get()
    if budget:
        budget.consume()
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", budget.max_output_tokens),
                                   budget.max_output_tokens)
        # Count textual context separately; image bytes have their own media cap.
        chars = sum(len(str(m.get("content", ""))) if isinstance(m.get("content"), str)
                    else sum(len(b.get("text", "")) for b in m.get("content", []))
                    for m in kwargs.get("messages", []))
        if chars > budget.max_input_chars:
            raise BudgetExceeded("api_input_limit")
    if kwargs.get("model") == "deepseek-chat":
        kwargs["model"] = text_model()
    kwargs.setdefault("max_tokens", int(setting("DEEPSEEK_MAX_OUTPUT_TOKENS", 2048)))
    return client.chat.completions.create(**kwargs)


class ManagedClient:
    def __init__(self, sdk=None):
        self._sdk = sdk
        self.records = []
        self.total_calls = 0
        self.chat = SimpleNamespace(completions=self)

    def _get_sdk(self):
        if self._sdk is None:
            key = setting("DEEPSEEK_API_KEY")
            if not key:
                raise APIError("missing_api_key")
            from openai import OpenAI
            self._sdk = OpenAI(
                api_key=key, base_url=setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout=float(setting("DEEPSEEK_TIMEOUT", 60)),
                max_retries=max(0, min(int(setting("DEEPSEEK_MAX_RETRIES", 2)), 3)),
            )
        return self._sdk

    def create(self, **kwargs):
        start = time.perf_counter()
        self.total_calls += 1
        model = kwargs.get("model", text_model())
        record = {"sequence": self.total_calls, "model": model, "ok": False, "prompt_tokens": 0,
                  "completion_tokens": 0, "estimated_cost_usd": None}
        try:
            response = self._get_sdk().chat.completions.create(**kwargs)
            if (not getattr(response, 'choices', None) or
                    not isinstance(response.choices[0].message.content, str) or
                    not response.choices[0].message.content.strip()):
                raise APIError('empty_response')
            record["ok"] = True
            record["response_model"] = str(getattr(response, "model", "") or model)
            usage = getattr(response, "usage", None)
            for field in ("prompt_tokens", "completion_tokens"):
                record[field] = int(getattr(usage, field, 0) or 0)
            record["total_tokens"] = int(getattr(usage, "total_tokens", 0) or
                                         (record["prompt_tokens"] + record["completion_tokens"]))
            try:
                rates = json.loads(setting("DEEPSEEK_PRICES_JSON", "{}")).get(model)
                if rates is not None:
                    record["estimated_cost_usd"] = (
                        record["prompt_tokens"] * float(rates["input"]) +
                        record["completion_tokens"] * float(rates["output"])) / 1_000_000
            except (ValueError, TypeError, KeyError):
                pass
            return response
        except APIError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            category = {401: "authentication", 403: "permission", 402: "balance",
                        429: "rate_limit", 400: "invalid_request"}.get(status, "connection_or_server")
            raise APIError(category) from None
        finally:
            record["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            self.records.append(record)
            del self.records[:-1000]
