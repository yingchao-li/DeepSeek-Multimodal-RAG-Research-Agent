"""Small, configurable safeguards for document-grounded answers."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
import uuid

from src.paths import APP_ROOT


def _number(name, default, minimum, maximum, cast=float):
    try:
        value = cast(os.environ.get(name, default))
        return value if minimum <= value <= maximum else default
    except (TypeError, ValueError):
        return default


def min_evidence_score():
    return _number('RAG_MIN_EVIDENCE_SCORE', 0.45, 0, 1)


def max_calls():
    return _number('RAG_MAX_CALLS', 8, 1, 30, int)


def question_limit():
    return _number('RAG_AUDIT_QUESTION_MAX_CHARS', 200, 0, 2000, int)


def _safe_question(question):
    value = str(question)[:question_limit()]
    value = re.sub(r'(?i)(DEEPSEEK_API_KEY\s*[:=]\s*)\S+', r'\1[REDACTED]', value)
    return re.sub(r'\bsk-[A-Za-z0-9_-]{8,}\b', '[REDACTED]', value)


def abstention_text(lang_code=None):
    from src.i18n import translate
    return translate('insufficient_evidence', lang_code or 'zh')


def is_abstention(answer):
    return any(answer == abstention_text(lang) for lang in ('zh', 'en', 'de', 'fr'))


def evidence_score(doc):
    if doc.get('grading_score') is not None:
        return max(0.0, min(1.0, float(doc['grading_score'])))
    if doc.get('evidence_score') is not None:
        return max(0.0, min(1.0, float(doc['evidence_score'])))
    if doc.get('score') is not None:
        return max(0.0, min(1.0, float(doc['score'])))
    # User-provided chunks without retrieval scores remain usable.
    return 1.0


def has_sufficient_evidence(docs):
    return bool(docs) and max(evidence_score(doc) for doc in docs) >= min_evidence_score()


_INJECTION_PATTERNS = (
    re.compile(r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions', re.I),
    re.compile(r'disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions', re.I),
    re.compile(r'(?:system|developer)\s+(?:prompt|message)\s*:', re.I),
    re.compile(r'忽略(?:之前|以上|所有).{0,12}(?:指令|规则|提示)', re.I),
)


def injection_flags(docs):
    return [{'chunk_id': doc.get('chunk_id'), 'pattern': f'pattern_{i+1}'}
            for doc in docs for i, pattern in enumerate(_INJECTION_PATTERNS)
            if pattern.search(doc.get('text') or '')]


_AUDIT = ContextVar('rag_audit', default=None)
_WRITE_LOCK = threading.Lock()
_FALLBACK_SESSION_ID = uuid.uuid4().hex


def current_audit():
    return _AUDIT.get()


class AuditState:
    def __init__(self, question, mode, client, session_id=None):
        self.question = _safe_question(question)
        self.mode = mode
        self.client = client
        self.session_id = str(session_id or getattr(client, 'audit_session_id', _FALLBACK_SESSION_ID))
        self.start = time.perf_counter()
        self.start_records = len(getattr(client, 'records', []))
        self.retrieved_chunk_ids = []
        self.evidence_scores = {}
        self.injection_flags = []
        self.abstained = False
        self.error = None

    def note_evidence(self, docs):
        for doc in docs:
            chunk_id = doc.get('chunk_id')
            if chunk_id and chunk_id not in self.retrieved_chunk_ids:
                self.retrieved_chunk_ids.append(chunk_id)
            if chunk_id:
                self.evidence_scores[chunk_id] = round(evidence_score(doc), 5)
        for flag in injection_flags(docs):
            if flag not in self.injection_flags:
                self.injection_flags.append(flag)

    def write(self):
        records = getattr(self.client, 'records', [])[self.start_records:]
        path = Path(os.environ.get('RAG_AUDIT_PATH', str(APP_ROOT / 'logs' / 'audit.jsonl')))
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': self.session_id,
            'question': self.question,
            'mode': self.mode,
            'retrieved_chunk_ids': self.retrieved_chunk_ids,
            'evidence_scores': self.evidence_scores,
            'abstained': self.abstained,
            'injection_flags': self.injection_flags,
            'token_usage': {
                'prompt_tokens': sum(int(r.get('prompt_tokens', 0) or 0) for r in records),
                'completion_tokens': sum(int(r.get('completion_tokens', 0) or 0) for r in records),
                'total_tokens': sum(int(r.get('total_tokens', 0) or 0) for r in records),
            },
            'api_calls': len(records),
            'latency_ms': round((time.perf_counter() - self.start) * 1000, 2),
            'error': self.error,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(payload, ensure_ascii=False) + '\n').encode('utf-8')
        with _WRITE_LOCK:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)


@contextmanager
def audit_request(question, mode, client, session_id=None):
    parent = current_audit()
    if parent is not None:
        yield parent
        return
    state = AuditState(question, mode, client, session_id)
    token = _AUDIT.set(state)
    try:
        yield state
    except Exception as exc:
        state.error = type(exc).__name__
        raise
    finally:
        try:
            state.write()
        except OSError:
            logging.getLogger(__name__).warning('audit_log_unavailable')
        finally:
            _AUDIT.reset(token)
