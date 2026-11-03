"""Small UI helpers extending the existing Streamlit page and i18n system."""
from __future__ import annotations
import json
import streamlit as st
from src.api import APIError, setting
from src.config import load_embedding_model
from src.embeddings import LIGHTWEIGHT_MODEL, QWEN_MODEL, ModelLoadError
from src.vector_store import RetrievalConfig, RetrievalIndex, build_retrieval_index, corpus_fingerprint
from src.agentic_rag import AgentLimits
from src.i18n import t
from src.media import MediaError, MAX_IMAGES, attachment_path, save_image
from src.citations import source_label


def sidebar_upgrade_settings():
    with st.expander(t('upgrade_settings'), expanded=False):
        strategy_labels = {s: t('retrieval_'+s) for s in ('dense', 'hybrid', 'hybrid_rerank')}
        strategy = st.selectbox(t('retrieval_strategy'), ['dense', 'hybrid', 'hybrid_rerank'],
                                format_func=lambda s: strategy_labels[s], key='retrieval_strategy')
        candidate_k = st.slider(t('candidate_k'), 5, 60, 20)
        rerank_k = st.slider(t('rerank_k'), 4, 40, 12)
        allow_download = st.checkbox(t('model_downloads'), value=False, help=t('model_download_help'))
        configured_model = setting('EMBEDDING_MODEL', LIGHTWEIGHT_MODEL)
        choices = list(dict.fromkeys([configured_model, LIGHTWEIGHT_MODEL, QWEN_MODEL]))
        model_name = st.selectbox(t('embedding_model'), choices)
        reranker = st.selectbox(t('reranker_model'), ['cross-encoder/ms-marco-MiniLM-L-6-v2',
                                                     'Qwen/Qwen3-Reranker-0.6B'])
        parser = st.selectbox(t('pdf_parser'), ['pymupdf', 'docling'], help=t('docling_help'))
        iterations = st.slider(t('agent_max_iter'), 1, 5, 2)
        tools = st.slider(t('agent_max_tools'), 0, 20, 8)
        calls = st.slider(t('agent_max_calls'), 2, 20, 8)
        tokens = st.select_slider(t('agent_max_output'), [512, 1024, 2048, 4096], value=1024)
        visuals = st.slider(t('agent_max_visuals'), 0, 4, 3)
        decompose = st.checkbox(t('agent_decompose'), value=True)
        web = st.checkbox(t('allow_agent_web'), value=True)
    embed_model = load_embedding_model(model_name, allow_download)
    previous = st.session_state.get('_embedding_fingerprint')
    if previous and previous != embed_model.fingerprint:
        st.info(t('index_reset'))
    st.session_state['_embedding_fingerprint'] = embed_model.fingerprint
    config = RetrievalConfig(strategy, candidate_k, rerank_k, reranker_model=reranker,
                             allow_download=allow_download)
    limits = AgentLimits(iterations, tools, calls, tokens, max_visuals=visuals)
    return embed_model, config, limits, parser, decompose, web


def ensure_index(index_key, chunks_key, embed_model):
    chunks = st.session_state.get(chunks_key) or []
    index = st.session_state.get(index_key)
    index = index_for_chunks(index, chunks, embed_model)
    st.session_state[index_key] = index
    return index


def index_for_chunks(index, chunks, embed_model):
    compatible = (isinstance(index, RetrievalIndex) and index.version == 1
                  and index.embedding_fingerprint == embed_model.fingerprint
                  and index.corpus_fingerprint == corpus_fingerprint(chunks))
    if chunks and not compatible:
        index = build_retrieval_index(chunks, embed_model)
    return index


def render_evidence(records):
    if records:
        with st.expander(t('evidence_title')):
            for record in records:
                st.write(f"[{record.get('citation', '')}] {source_label(record)}")


def render_message(message):
    st.write(message.get('content', ''))
    for attachment in message.get('attachments') or []:
        try:
            st.image(str(attachment_path(attachment)), caption=attachment.get('name'), width=400)
        except MediaError as exc:
            st.warning(t(str(exc)))
    render_evidence(message.get('evidence'))


def image_upload(chat_id):
    epoch_key = '_image_epoch_'+str(chat_id)
    epoch = st.session_state.get(epoch_key, 0)
    files = st.file_uploader(t('image_upload'), type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
                             accept_multiple_files=True, help=t('image_help'),
                             key=f'vision_{chat_id}_{epoch}')
    valid, error = [], False
    if len(files) > MAX_IMAGES:
        st.error(t('image_count'))
        error = True
    else:
        for file in files:
            try:
                image = save_image(file.getvalue(), file.name)
                valid.append(image)
                st.image(str(attachment_path(image)), caption=file.name, width=200)
            except MediaError as exc:
                st.error(t(str(exc)))
                error = True
    send = st.button(t('image_send'), disabled=not valid or error, key=f'send_image_{chat_id}_{epoch}')
    return valid, send, error, epoch_key


def show_error(exc):
    if isinstance(exc, ModelLoadError):
        return t('embedding_unavailable')
    if isinstance(exc, MediaError):
        return t(str(exc))
    if isinstance(exc, APIError):
        return t('missing_api_key') if str(exc) == 'missing_api_key' else t('api_error', reason=t(str(exc)))
    # Never expose exceptions that could carry request bodies or credentials.
    return t('processing_error', reason=type(exc).__name__)


def retrieval_feedback(docs):
    if any(d.get('retrieval', {}).get('warning') == 'reranker_unavailable' for d in docs):
        st.warning(t('reranker_unavailable'))


def render_usage(client):
    records = client.records
    st.caption(t('usage_title'))
    st.write(f"{t('usage_calls')}: {len(records)}")
    inputs = sum(r['prompt_tokens'] for r in records)
    outputs = sum(r['completion_tokens'] for r in records)
    st.write(f"{t('usage_tokens')}: {inputs} / {outputs}")
    if records and all(r.get('estimated_cost_usd') is not None and r['ok'] for r in records):
        st.caption(t('usage_cost', cost=f"{sum(r['estimated_cost_usd'] for r in records):.6f}"))
    else:
        st.caption(t('usage_cost_unknown'))
    st.caption(t('usage_caption'))


def evaluation_panel(embed_model, config):
    from src.evaluation import run_benchmark, save_report
    st.markdown(f"### {t('eval_title')}")
    st.caption(t('eval_caption'))
    k = st.slider(t('eval_k'), 1, 8, 5)
    if st.button(t('eval_run')):
        try:
            with st.spinner(t('generating')):
                report = run_benchmark(embed_model=embed_model, k=k, config=config)
                save_report(report)
                st.session_state['_evaluation_report'] = report
        except Exception as exc:
            st.error(t('eval_failed')+' '+type(exc).__name__)
    report = st.session_state.get('_evaluation_report')
    if report:
        import pandas as pd
        import plotly.express as px
        st.dataframe(pd.DataFrame(report['summary']), use_container_width=True)
        measured = [r for r in report['summary'] if r.get('status') == 'measured']
        if measured:
            frame = pd.DataFrame(measured).melt(id_vars=['configuration'],
                    value_vars=['recall_at_k', 'precision_at_k', 'mrr', 'ndcg_at_k'])
            st.plotly_chart(px.bar(frame, x='configuration', y='value', color='variable', barmode='group'),
                            use_container_width=True)
        st.download_button(t('eval_download'), json.dumps(report, ensure_ascii=False, indent=2),
                           file_name='retrieval-evaluation.json', mime='application/json')
