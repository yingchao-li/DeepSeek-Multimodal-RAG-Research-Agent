"""
General-chat web helpers: rewrite the user query for search engines and format this session's web-memory for the LLM.
(App and chat modules perform actual fetching and persistence.)
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.prompts import build_history_block

MODEL = "deepseek-chat"


from src.api import request


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(text[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def refine_search_query(query: str, history: list[dict], client) -> str:
    """把用户问题改写成更适合搜索引擎的短查询。"""
    q = (query or "").strip()
    if not q:
        return q

    system = (
        "将用户当前问题改写成适合网页搜索的简短查询（可中英文），用于获取最新资讯与事实。\n"
        "只输出 JSON：{\"q\":\"...\"}"
    )
    user = f"最近对话：\n{build_history_block(history)}\n\n用户问题：\n{q}"

    try:
        r = request(client,
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        raw = (r.choices[0].message.content or "").strip()
        data = _parse_json_object(raw) or {}
        out = (data.get("q") or data.get("query") or "").strip()
        return out if out else q
    except Exception:
        return q


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def build_supplementary_context(
    web_memory: list[dict],
    *,
    max_records: int = 12,
    max_item_chars: int = 450,
) -> str:
    """将本会话已保存的网页检索记录格式化为模型可读文本。"""
    if not web_memory:
        return ""

    lines: list[str] = []
    recent = web_memory[-max_records:]
    lines.append("【本会话已保存的网页检索记忆（按时间顺序）】")

    for rec in recent:
        at = rec.get("at", "")
        sq = rec.get("search_query", "")
        lines.append(f"- 时间 {at} | 检索词: {sq}")
        for it in rec.get("items") or []:
            src = it.get("source", "")
            if it.get('url'):
                src += ' ('+str(it['url'])+')'
            body = _truncate(it.get("text", ""), max_item_chars)
            lines.append(f"  · {src}: {body}")
        lines.append("")

    return "\n".join(lines).strip()
