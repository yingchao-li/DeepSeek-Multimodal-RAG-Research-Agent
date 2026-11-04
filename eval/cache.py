"""Append-only JSONL cache so paid evaluation can resume after interruption."""
from __future__ import annotations
import json
from pathlib import Path


class ResultCache:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    self.rows[row["cache_key"]] = row["value"]
                except (ValueError, KeyError):
                    continue

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"cache_key": key, "value": value}, ensure_ascii=False) + "\n")
        self.rows[key] = value

