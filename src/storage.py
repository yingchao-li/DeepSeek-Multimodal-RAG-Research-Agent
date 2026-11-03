"""
JSON file persistence for chat sessions (messages, titles, timestamps, web memory) under data/chats.json.
"""

import json
import os
import tempfile
from src.paths import DATA_DIR


CHAT_FILE = DATA_DIR / "chats.json"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_chats_to_disk(chats: dict):
    ensure_data_dir()

    fd, temporary = tempfile.mkstemp(prefix="chats-", suffix=".tmp", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, CHAT_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_chats_from_disk() -> dict:
    ensure_data_dir()

    if not CHAT_FILE.exists():
        return {}

    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def delete_chat_file():
    if CHAT_FILE.exists():
        CHAT_FILE.unlink()
