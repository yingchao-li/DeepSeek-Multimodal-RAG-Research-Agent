"""
Research-paper summarization and full-text Q&A: prompts and DeepSeek calls with reply-language directives matching the UI.
"""

from src.i18n import reply_language_directive


from src.api import request


def build_paper_summary_prompt(paper_text: str) -> str:
    shortened = paper_text[:20000]

    return f"""Based on the following paper text, produce a structured summary:

Requirements:
1) Start with a one-sentence overview.
2) Then bullet sections for: research question, core method, main contributions, experiments/results, limitations.
3) If the paper does not support a point, state that clearly.

Paper text:
{shortened}
"""


def summarize_paper(paper_text: str, client, lang_code: str = "zh") -> str:
    prompt = build_paper_summary_prompt(paper_text)
    system = (
        "You are a rigorous research-paper analysis assistant.\n\n"
        + reply_language_directive(lang_code)
    )

    response = request(client,
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def build_paper_chat_prompt(paper_text: str, query: str, history: list[dict]) -> str:
    shortened_paper = paper_text[:25000]

    history_text = ""
    if history:
        lines = []
        for msg in history:
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(lines)
    else:
        history_text = "无"

    return f"""
Answer using the paper content below. Rules:
1) Prefer evidence from the paper.
2) If not stated in the paper, say clearly that it is not stated.
3) Use recent chat history for follow-up questions.

Paper content:
{shortened_paper}

Recent chat history:
{history_text}

Current question:
{query}
"""


def chat_with_paper(
    paper_text: str,
    query: str,
    history: list[dict],
    client,
    lang_code: str = "zh",
) -> str:
    prompt = build_paper_chat_prompt(
        paper_text=paper_text,
        query=query,
        history=history,
    )
    system = (
        "You are a rigorous research-paper Q&A assistant.\n\n"
        + reply_language_directive(lang_code)
    )

    response = request(client,
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content
