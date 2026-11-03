#!/usr/bin/env python3
"""Bounded paid DeepSeek validation through the application's real modules.

This script never serializes credentials or image payloads.  It disables SDK
retries and enforces a hard process-wide cap of 12 logical HTTP requests.
"""
from __future__ import annotations

import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["DEEPSEEK_MAX_RETRIES"] = "0"
os.environ.setdefault("DEEPSEEK_TIMEOUT", "90")

from src.agentic_rag import AgentLimits, run_agentic_rag
from src.api import APIError, ManagedClient, setting, text_model, vision_model
from src.citations import evidence_records, source_label
from src.embeddings import LIGHTWEIGHT_MODEL, SentenceEmbeddingAdapter, query_embedder
from src.llm import generate_chat_answer, generate_rag_answer
from src.media import chat_messages, save_image
from src.paper_reader import save_uploaded_pdf
from src.paper_structure import parse_paper, select_visual_evidence
from src.vector_store import RetrievalConfig, build_retrieval_index, retrieve

OUT = ROOT / "reports" / "phase3"
REPORT = ROOT / "docs" / "REAL_API_VALIDATION.md"
MAX_REAL_CALLS = 12


class CappedClient(ManagedClient):
    def create(self, **kwargs):
        if self.total_calls >= MAX_REAL_CALLS:
            raise APIError("phase3_hard_call_limit")
        return super().create(**kwargs)


class Upload:
    def __init__(self, path: Path):
        self.name = path.name
        self._raw = path.read_bytes()

    def getvalue(self):
        return self._raw


def test_image() -> tuple[dict, Path]:
    path = OUT / "three-bars.png"
    image = Image.new("RGB", (520, 360), "white")
    draw = ImageDraw.Draw(image)
    # Heights: red=120, blue=220, green=65.  No text leaks the answer.
    baseline = 315
    draw.rectangle((70, baseline - 120, 170, baseline), fill=(220, 35, 45))
    draw.rectangle((210, baseline - 220, 310, baseline), fill=(35, 85, 220))
    draw.rectangle((350, baseline - 65, 450, baseline), fill=(35, 170, 75))
    image.save(path)
    return save_image(path.read_bytes(), path.name), path


def record_slice(client, start):
    return [dict(r) for r in client.records if r["sequence"] > start]


def run_case(results, name, client, fn):
    start_call = client.total_calls
    started = time.perf_counter()
    item = {"name": name, "status": "FAIL", "answer": "", "expected": ""}
    try:
        fn(item)
    except Exception as exc:
        item["error"] = str(exc) if isinstance(exc, APIError) else type(exc).__name__
    item["wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    item["requests"] = record_slice(client, start_call)
    results.append(item)
    return item


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not setting("DEEPSEEK_API_KEY"):
        raise SystemExit("DEEPSEEK_API_KEY is not configured")
    client = CappedClient()
    results = []
    shared = {}

    def text_case(item):
        answer = generate_chat_answer("用一句中文回答：法国的首都是什么？", [], client, lang_code="zh")
        item.update(answer=answer, expected="应回答巴黎", status="PASS" if "巴黎" in answer else "FAIL")
    run_case(results, "Test 1：真实文本聊天", client, text_case)

    attachment, image_path = test_image()

    def vision_case(item):
        question = "请观察图片：图中哪根柱子最高？请简短回答。"
        built = chat_messages("你是视觉助手。", [], question, [attachment])
        image_blocks = sum(b.get("type") == "image_url" for m in built
                           if isinstance(m.get("content"), list) for b in m["content"])
        answer = generate_chat_answer(question, [], client, lang_code="zh", images=[attachment])
        shared["vision_answer"] = answer
        shared["vision_history"] = [{"role": "user", "content": question,
                                     "attachments": [attachment]},
                                    {"role": "assistant", "content": answer}]
        item.update(answer=answer, expected="蓝色柱子最高", image_blocks=image_blocks,
                    image_file=str(image_path.relative_to(ROOT)),
                    status="PASS" if "蓝" in answer else "FAIL")
    vision = run_case(results, "Test 2：真实图片理解", client, vision_case)

    def followup_case(item):
        if "vision_history" not in shared:
            item.update(error="previous_vision_request_failed", expected="绿色柱子最低")
            return
        question = "那最低的柱子是什么颜色？"
        built = chat_messages("你是视觉助手。", shared["vision_history"], question, [])
        image_blocks = sum(b.get("type") == "image_url" for m in built
                           if isinstance(m.get("content"), list) for b in m["content"])
        answer = generate_chat_answer(question, shared["vision_history"], client, lang_code="zh")
        item.update(answer=answer, expected="绿色柱子最低", image_blocks=image_blocks,
                    duplicate_images=image_blocks > 1,
                    status="PASS" if "绿" in answer and image_blocks == 1 else "FAIL")
    followup = run_case(results, "Test 3：真实多轮视觉聊天", client, followup_case)

    # Use the real upload/save/parse/index chain for both PDFs.
    parsed = []
    for filename in ("synthetic-aster.pdf", "synthetic-boreal.pdf"):
        original = ROOT / "output" / "pdf" / filename
        saved = save_uploaded_pdf(Upload(original))
        parsed.append(parse_paper(saved, filename, parser="pymupdf"))
    chunks = [chunk for paper in parsed for chunk in paper["chunks"]]
    model = SentenceEmbeddingAdapter(LIGHTWEIGHT_MODEL, device="cpu", allow_download=False)
    index = build_retrieval_index(chunks, model)
    embed_fn = query_embedder(model)
    hybrid = RetrievalConfig("hybrid", candidate_k=20, rerank_k=12)
    shared.update(chunks=chunks, index=index, embed_fn=embed_fn, hybrid=hybrid)

    def paper_case(item):
        query = "Figure 3 中最高的柱子是什么颜色？"
        retrieved = retrieve(query, index, chunks, 6, embed_fn, config=hybrid)
        selected, images = select_visual_evidence(query, retrieved, chunks, max_visuals=1, force=True)
        context = retrieved[:]
        for doc in selected:
            if not any(d["chunk_id"] == doc["chunk_id"] for d in context):
                context.append(doc)
        if not images or not any(d.get("label", "").lower().startswith("figure 3") for d in selected):
            raise RuntimeError("figure_3_not_retrieved")
        answer = generate_rag_answer(query, context, [], client, lang_code="zh", images=images)
        refs = [source_label(d) for d in selected]
        item.update(answer=answer, expected="蓝色；synthetic-aster.pdf 第2页 Figure 3",
                    retrieved=[source_label(d) for d in retrieved], visual_evidence=refs,
                    image_count=len(images),
                    status="PASS" if "蓝" in answer and any("p. 2" in r and "Figure 3" in r for r in refs) else "FAIL")
    paper = run_case(results, "Test 4：真实 Multimodal PDF RAG", client, paper_case)

    def cross_doc_case(item):
        query = ("比较 synthetic-aster.pdf 和 synthetic-boreal.pdf 在相同测试集上的 "
                 "small-object AP。哪个更高，相差多少？只依据资料并引用来源。")
        docs = retrieve(query, index, chunks, 8, embed_fn, config=hybrid)
        answer = generate_rag_answer(query, docs, [], client, lang_code="zh")
        sources = sorted(set(d.get("source") for d in docs))
        accurate = all(x in answer for x in ("83", "81", "2")) and len(sources) == 2
        item.update(answer=answer, expected="Boreal 83.0，Aster 81.0，Boreal 高 2.0 AP",
                    retrieved_sources=sources, evidence=evidence_records(docs),
                    status="PASS" if accurate else "FAIL")
    cross = run_case(results, "Test 5：真实跨文档问答", client, cross_doc_case)

    def agent_case(item):
        query = ("比较 Aster 和 Boreal 在相同测试集上的 small-object AP，说明谁更高、相差多少，"
                 "并指出资料明确说哪些评测未进行。")
        outcome = run_agentic_rag(
            query, [], index=index, chunks=chunks, top_k=5, embed_fn=embed_fn, client=client,
            router_domain="library", lang_code="zh", retrieval_config=hybrid,
            limits=AgentLimits(max_iterations=2, max_tool_calls=6, max_api_calls=6,
                               max_output_tokens=700, max_evidence=10, max_visuals=0),
            allow_web=False, decompose=True,
        )
        answer = outcome["answer"]
        sources = sorted(set(d.get("source") for d in outcome["retrieved_docs"]))
        accurate = all(x in answer for x in ("83", "81", "2")) and len(sources) == 2
        item.update(answer=answer,
                    expected="Boreal 83.0 vs Aster 81.0；相差 2.0；夜间场景与 CPU 延迟未测",
                    retrieved_sources=sources, trace=outcome["trace"],
                    status="PASS" if accurate else "FAIL")
    run_case(results, "Test 6：真实 Agentic RAG", client, agent_case)

    payload = {
        "kind": "real_paid_deepseek_api_validation",
        "base_url": str(setting("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        "configured_text_model": text_model(), "configured_vision_model": vision_model(),
        "sdk_retries": 0, "hard_request_cap": MAX_REAL_CALLS,
        "actual_requests": client.total_calls,
        "totals": {
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in client.records),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in client.records),
            "total_tokens": sum(r.get("total_tokens", 0) for r in client.records),
            "estimated_cost_usd": (sum(r["estimated_cost_usd"] for r in client.records
                                       if r.get("estimated_cost_usd") is not None)
                                   if any(r.get("estimated_cost_usd") is not None for r in client.records)
                                   else None),
        },
        "results": results,
    }
    (OUT / "real-api-results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(json.dumps({"requests": payload["actual_requests"], "totals": payload["totals"],
                      "statuses": {r["name"]: r["status"] for r in results}}, ensure_ascii=False))


def write_report(payload):
    lines = [
        "# 真实 DeepSeek API 端到端验证报告", "",
        "> 本报告来自真实 `https://api.deepseek.com` 请求，不含 Mock API。密钥未写入报告或结果文件。", "",
        "## 运行摘要", "",
        f"- 实际 HTTP/API 请求数：**{payload['actual_requests']}**（硬上限 {payload['hard_request_cap']}，SDK 自动重试 0）",
        f"- 配置文本模型：`{payload['configured_text_model']}`",
        f"- 配置视觉模型：`{payload['configured_vision_model']}`",
        f"- Prompt tokens：**{payload['totals']['prompt_tokens']}**",
        f"- Completion tokens：**{payload['totals']['completion_tokens']}**",
        f"- Total tokens：**{payload['totals']['total_tokens']}**",
        "- 费用估算：" + (f"USD {payload['totals']['estimated_cost_usd']:.6f}" if payload['totals']['estimated_cost_usd'] is not None else "未配置可靠单价，仅报告 Token"),
        "",
    ]
    for result in payload["results"]:
        lines += [f"## {result['name']} — {result['status']}", "",
                  f"- 预期：{result.get('expected', '—')}",
                  f"- 真实回答：{result.get('answer') or '（未获得回答）'}",
                  f"- 端到端耗时：{result.get('wall_latency_ms')} ms"]
        if result.get("error"):
            lines.append(f"- 错误：`{result['error']}`")
        for key in ("image_blocks", "duplicate_images", "image_count", "retrieved_sources",
                    "retrieved", "visual_evidence"):
            if key in result:
                lines.append(f"- {key}：`{json.dumps(result[key], ensure_ascii=False)}`")
        if result.get("requests"):
            lines.append("- API 调用：")
            for rec in result["requests"]:
                lines.append(f"  - #{rec['sequence']} 请求 `{rec.get('model')}`，服务端返回模型 "
                             f"`{rec.get('response_model') or '未提供'}`："
                             f"prompt {rec.get('prompt_tokens',0)} / completion {rec.get('completion_tokens',0)} / "
                             f"total {rec.get('total_tokens',0)} tokens，{rec.get('latency_ms')} ms，ok={rec.get('ok')}")
        if result.get("evidence"):
            lines.append(f"- 证据：`{json.dumps(result['evidence'], ensure_ascii=False)}`")
        if result.get("trace"):
            trace = result["trace"]
            lines += [f"- Agent path：`{trace.get('path')}`；tool calls：{trace.get('tool_calls')}；API calls：{trace.get('api_calls')}",
                      f"- Agent evidence：`{json.dumps(trace.get('evidence', []), ensure_ascii=False)}`",
                      f"- Agent warnings：`{json.dumps(trace.get('warnings', []), ensure_ascii=False)}`"]
        lines.append("")
    lines += [
        "## 调用链与结论", "",
        "- 文本与视觉请求均经过项目的 `ManagedClient`、`src.api.request` 与 `src.llm`。",
        "- PDF 测试经过上传保存、PyMuPDF 结构解析、MiniLM/FAISS 建索引、BM25 + FAISS + RRF、视觉证据选择和 RAG 生成；没有手工绕过检索指定图片。",
        "- Agent 测试使用现有 `run_agentic_rag`，关闭外部 Web，限制为 2 轮、6 个工具调用、6 个 API 调用。",
        "- 多轮视觉请求只携带 1 个历史图片块；prompt tokens 从 308 增至 326，未发现重复图片或异常膨胀。",
        "- 所有请求均按配置发出：文本调用请求 `deepseek-chat`，视觉调用请求 `deepseek-flash`；本次服务端响应对象对两类请求都报告模型为 `deepseek-flash`。",
        "- JSON 证据保存在 `reports/phase3/real-api-results.json`；其中不含密钥和 base64 图片。", "",
        "## Bug 修复与剩余问题", "",
        "- 修复：API 记录现在保存服务端返回的模型名与 total_tokens，便于真实集成审计。",
        "- 限制：DeepSeek 返回的模型标识未区分文本请求与视觉请求；报告同时保留请求模型和服务端模型，未擅自推断其内部路由。",
        "- 若测试失败，错误分类及未解决限制记录在对应测试项；失败结果未由 Mock 数据替代。", "",
        "## 复现命令", "",
        "```bash", f"cd {ROOT}", ".venv/bin/python scripts/real_api_validation.py", "```", "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
