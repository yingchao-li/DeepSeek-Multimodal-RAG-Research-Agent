"""
Multi-chat session management for Streamlit: create/switch/delete chats, append messages,
recent dialogue history for the LLM, optional per-session web-search memory, and disk sync via storage.
"""

import streamlit as st
from datetime import datetime
import uuid

from src.storage import load_chats_from_disk, save_chats_to_disk


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def persist_chats():
    chats = st.session_state.get("chats", {})
    save_chats_to_disk(chats)


def _ensure_web_memory_on_chats(chats: dict) -> None:
    for c in chats.values():
        if "web_memory" not in c:
            c["web_memory"] = []


def init_chat_state():
    if "chats" not in st.session_state:
        st.session_state["chats"] = load_chats_from_disk()
        _ensure_web_memory_on_chats(st.session_state["chats"])

    if "current_chat_id" not in st.session_state:
        # 默认选最近更新的聊天
        chats = st.session_state["chats"]
        if chats:
            sorted_chats = sorted(
                chats.values(),
                key=lambda x: x.get("updated_at", ""),
                reverse=True
            )
            st.session_state["current_chat_id"] = sorted_chats[0]["id"]
        else:
            st.session_state["current_chat_id"] = None


def create_new_chat(title: str = "新聊天") -> str:
    chat_id = str(uuid.uuid4())
    timestamp = now_str()

    st.session_state["chats"][chat_id] = {
        "id": chat_id,
        "title": title,
        "messages": [],
        "web_memory": [],
        "created_at": timestamp,
        "updated_at": timestamp,
    }

    st.session_state["current_chat_id"] = chat_id
    persist_chats()
    return chat_id


def get_current_chat_id():
    return st.session_state.get("current_chat_id")


def set_current_chat(chat_id: str):
    if chat_id in st.session_state["chats"]:
        st.session_state["current_chat_id"] = chat_id


def get_current_chat():
    chat_id = get_current_chat_id()
    if not chat_id:
        return None
    return st.session_state["chats"].get(chat_id)


def get_current_messages():
    chat = get_current_chat()
    if not chat:
        return []
    return chat["messages"]


def get_current_web_memory() -> list[dict]:
    chat = get_current_chat()
    if not chat:
        return []
    if "web_memory" not in chat:
        chat["web_memory"] = []
    return chat["web_memory"]


def append_web_memory_record(search_query: str, items: list[dict], max_entries: int = 25) -> None:
    """将一轮网页检索结果写入当前会话记忆并落盘。"""
    chat = get_current_chat()
    if chat is None:
        return
    if "web_memory" not in chat:
        chat["web_memory"] = []

    chat["web_memory"].append({
        "at": now_str(),
        "search_query": (search_query or "").strip(),
        "items": items,
    })
    while len(chat["web_memory"]) > max_entries:
        chat["web_memory"].pop(0)

    chat["updated_at"] = now_str()
    persist_chats()


def clear_current_web_memory() -> None:
    chat = get_current_chat()
    if chat:
        chat["web_memory"] = []
        chat["updated_at"] = now_str()
        persist_chats()


def update_chat_title(chat_id: str, title: str):
    if chat_id in st.session_state["chats"]:
        st.session_state["chats"][chat_id]["title"] = title
        st.session_state["chats"][chat_id]["updated_at"] = now_str()
        persist_chats()


def add_message(role: str, content: str, *, attachments=None, evidence=None):
    chat = get_current_chat()
    if chat is None:
        chat_id = create_new_chat()
        chat = st.session_state["chats"][chat_id]

    message = {
        "role": role,
        "content": content
    }
    if attachments and role == "user":
        from src.media import attachment_path
        for attachment in attachments:
            attachment_path(attachment)
        message["attachments"] = [
            {k: a[k] for k in ("id", "name", "path", "mime", "width", "height") if k in a}
            for a in attachments
        ]
    if evidence:
        message["evidence"] = evidence
    chat["messages"].append(message)
    chat["updated_at"] = now_str()

    if role == "user" and (chat["title"] == "新聊天" or not chat["title"].strip()):
        new_title = content.strip().replace("\n", " ")[:25]
        if new_title:
            chat["title"] = new_title

    persist_chats()


def clear_current_chat_messages():
    chat = get_current_chat()
    if chat:
        chat["messages"] = []
        chat["web_memory"] = []
        chat["updated_at"] = now_str()
        persist_chats()


def delete_chat(chat_id: str):
    if chat_id in st.session_state["chats"]:
        del st.session_state["chats"][chat_id]

        # 如果删的是当前会话，自动切到最近更新的一个
        if st.session_state.get("current_chat_id") == chat_id:
            chats = st.session_state["chats"]
            if chats:
                sorted_chats = sorted(
                    chats.values(),
                    key=lambda x: x.get("updated_at", ""),
                    reverse=True
                )
                st.session_state["current_chat_id"] = sorted_chats[0]["id"]
            else:
                st.session_state["current_chat_id"] = None

        persist_chats()


def get_recent_history(max_turns: int = 3, exclude_current_user: bool = True):
    messages = get_current_messages()
    if exclude_current_user and messages and messages[-1].get("role") == "user":
        messages = messages[:-1]
    max_messages = max_turns * 2
    recent = messages[-max_messages:]
    if not any(m.get("attachments") for m in recent):
        previous = next((m for m in reversed(messages[:-max_messages])
                         if m.get("role") == "user" and m.get("attachments")), None)
        if previous:
            recent = [previous] + recent
    return recent


def search_chats(keyword: str):
    chats = st.session_state.get("chats", {})
    keyword = keyword.strip().lower()

    if not keyword:
        return list(chats.values())

    results = []
    for chat in chats.values():
        title = chat["title"].lower()
        joined_messages = " ".join([m["content"] for m in chat["messages"]]).lower()
        mem_parts = []
        for rec in chat.get("web_memory") or []:
            mem_parts.append(str(rec.get("search_query", "")))
            for it in rec.get("items") or []:
                mem_parts.append(str(it.get("source", "")))
                mem_parts.append(str(it.get("text", ""))[:200])
        joined_memory = " ".join(mem_parts).lower()

        if keyword in title or keyword in joined_messages or keyword in joined_memory:
            results.append(chat)

    results.sort(key=lambda x: x["updated_at"], reverse=True)
    return results


def get_recent_chats(limit: int = 10):
    chats = list(st.session_state.get("chats", {}).values())
    chats.sort(key=lambda x: x["updated_at"], reverse=True)
    return chats[:limit]
