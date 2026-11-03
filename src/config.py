"""
Load and cache the sentence-transformer embedding model and OpenAI-compatible DeepSeek API client from Streamlit secrets.
"""

import streamlit as st
from src.api import ManagedClient, setting
from src.embeddings import SentenceEmbeddingAdapter


@st.cache_resource(max_entries=2)
def _embedding_resource(model_name,device):
    return SentenceEmbeddingAdapter(model_name,device=device)


def load_embedding_model(model_name=None, allow_download=False):
    adapter = _embedding_resource(model_name or setting('EMBEDDING_MODEL','all-MiniLM-L6-v2'),
                                  setting('MODEL_DEVICE','cpu'))
    # Download permission is a loading policy, not a second copy of the model.
    if adapter._model is None:
        adapter.allow_download = allow_download
    return adapter


def load_llm_client():
    if "_managed_llm_client" not in st.session_state:
        st.session_state["_managed_llm_client"] = ManagedClient()
    return st.session_state["_managed_llm_client"]
