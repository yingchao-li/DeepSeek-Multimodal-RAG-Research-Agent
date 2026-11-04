"""
Thin wrappers around the DeepSeek chat API: retrieval-grounded answers (RAG) and general multi-turn chat,
with optional supplementary web context and UI-aligned reply-language instructions.
"""

from contextlib import nullcontext
from typing import Optional

from src.i18n import reply_language_directive
from src.prompts import build_system_prompt, build_user_prompt
from src.api import api_budget, current_budget, request, vision_model
from src.citations import has_valid_citation, validate_citations
from src.guardrails import (abstention_text, audit_request, has_sufficient_evidence,
                            max_calls)
from src.media import chat_messages, contains_images, user_content, bind_image_citations



def generate_rag_answer(
    query: str,
    retrieved_docs: list[dict],
    history: list[dict],
    client,
    *,
    lang_code: Optional[str] = None,
    images: Optional[list[dict]] = None,
    strict_citations: bool = False,
) -> str:
    with audit_request(query, 'plain_rag', client) as audit:
        audit.note_evidence(retrieved_docs)
        if not has_sufficient_evidence(retrieved_docs):
            audit.abstained = True
            return abstention_text(lang_code)
        system_prompt = build_system_prompt(lang_code)
        user_prompt = build_user_prompt(query=query, retrieved_docs=retrieved_docs, history=history)
        budget = nullcontext() if current_budget() else api_budget(max_calls=max_calls())
        with budget:
            for attempt in range(1 if strict_citations else 2):
                instruction = ('' if attempt == 0 and not strict_citations else
                               '\nYour previous answer lacked a valid [S1] citation. '
                               'Cite at least one provided source label for each supported conclusion. '
                               'Do not invent citations.')
                response = request(
                    client, model=vision_model() if images else 'deepseek-chat',
                    messages=[
                        {'role': 'system', 'content': system_prompt + instruction},
                        {'role': 'user', 'content': user_content(
                            user_prompt, bind_image_citations(images, retrieved_docs))},
                    ], temperature=0.2,
                )
                answer, _ = validate_citations(response.choices[0].message.content, retrieved_docs)
                if has_valid_citation(answer, retrieved_docs):
                    return answer
        audit.abstained = True
        return abstention_text(lang_code)


def generate_chat_answer(
    query: str,
    history: list[dict],
    client,
    *,
    supplementary_context: Optional[str] = None,
    lang_code: Optional[str] = None,
    images: Optional[list[dict]] = None,
) -> str:
    system_prompt = (
        "你是一个有帮助的 AI 助手。"
        "请结合最近聊天历史，清晰、自然地回答用户问题。"
        "如果用户的问题不明确，可以根据上下文合理理解。"
    )
    if lang_code:
        system_prompt += "\n\n" + reply_language_directive(lang_code)
    if supplementary_context and supplementary_context.strip():
        system_prompt += (
            "\n\n【以下为系统提供的补充资料：本会话已保存的网页检索记忆，以及本轮检索结果。"
            "若与用户问题相关请优先参考并说明信息可能有时效性；若为闲聊可忽略。】\n"
            + supplementary_context.strip()
        )

    messages = chat_messages(system_prompt, history, query, images)

    with audit_request(query, 'general_chat', client):
        budget = nullcontext() if current_budget() else api_budget(max_calls=max_calls())
        with budget:
            response = request(client,
                model=vision_model() if contains_images(messages) else "deepseek-chat",
                messages=messages,
                temperature=0.5,
            )
        return response.choices[0].message.content
