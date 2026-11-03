"""
Batch-encode text strings into L2-normalized float32 vectors using a sentence-transformers model.
"""

from __future__ import annotations
import hashlib
import json
import threading
from functools import lru_cache
import numpy as np

LIGHTWEIGHT_MODEL = "all-MiniLM-L6-v2"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class ModelLoadError(RuntimeError):
    pass


class SentenceEmbeddingAdapter:
    def __init__(self, model_name=LIGHTWEIGHT_MODEL, device="cpu", revision=None, allow_download=False):
        self.model_name, self.device, self.revision = model_name, device, revision
        self._model = None
        self.allow_download = allow_download
        self._lock = threading.Lock()

    @property
    def fingerprint(self):
        payload = {"model": self.model_name, "revision": self.revision,
                   "normalize": True, "adapter_version": 2,
                   "dtype_policy": "cpu_fp32_else_checkpoint_auto" if self.model_name == QWEN_MODEL else 'default',
                   "query_prompt": "query" if self.model_name == QWEN_MODEL else None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    kwargs = {"device": self.device}
                    if self.model_name != LIGHTWEIGHT_MODEL:
                        kwargs['local_files_only'] = not self.allow_download
                    if self.model_name == QWEN_MODEL:
                        import torch
                        kwargs['model_kwargs'] = {'torch_dtype': torch.float32 if self.device == 'cpu' else 'auto'}
                    if self.revision:
                        kwargs["revision"] = self.revision
                    try:
                        self._model = SentenceTransformer(self.model_name, **kwargs)
                    except Exception:
                        raise ModelLoadError('embedding_unavailable') from None
        return self._model

    def encode(self, texts, *, is_query=False):
        kwargs = {"prompt_name": "query"} if is_query and self.model_name == QWEN_MODEL else {}
        return self._load().encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                   show_progress_bar=False, **kwargs).astype("float32")


def embed_texts(texts: list[str], embed_model, *, is_query=False) -> np.ndarray:
    if isinstance(embed_model, SentenceEmbeddingAdapter):
        return embed_model.encode(texts, is_query=is_query)
    vectors = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vectors.astype("float32")


def query_embedder(model, cache_size=64):
    @lru_cache(maxsize=cache_size)
    def cached(texts, fingerprint):
        return embed_texts(list(texts), model, is_query=True)
    def encode(texts):
        return cached(tuple(texts), getattr(model, 'fingerprint', None)).copy()
    encode.fingerprint = getattr(model, "fingerprint", None)
    encode.cache_info = cached.cache_info
    return encode
