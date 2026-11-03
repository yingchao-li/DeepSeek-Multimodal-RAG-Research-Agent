"""Citations come only from retrieved metadata; never manufacture page numbers."""
from __future__ import annotations
import re


def source_label(doc):
    label = str(doc.get("source", "unknown"))
    if doc.get("page") is not None:
        label += f", p. {doc['page']}"
    if doc.get("label"):
        label += f", {doc['label']}"
    if doc.get("section"):
        label += f", {doc['section']}"
    if doc.get("url"):
        label += f" ({doc['url']})"
    return label


def evidence_block(docs):
    return "\n\n".join(f"[S{i}] {source_label(d)}\n{d.get('text', '')}"
                       for i, d in enumerate(docs, 1))


def validate_citations(answer, docs):
    invalid = []
    def check(match):
        if 1 <= int(match.group(1)) <= len(docs):
            return match.group(0)
        invalid.append(match.group(0))
        return ""
    return re.sub(r"\[S(\d+)\]", check, answer), invalid


def evidence_records(docs):
    keys = ("chunk_id", "source", "page", "section", "label", "kind", "url")
    return [{"citation": f"S{i}", **{k: d[k] for k in keys if d.get(k) is not None}}
            for i, d in enumerate(docs, 1)]
