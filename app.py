"""
Streamlit entry point: multi-mode app (document RAG, paper reading, tabular data analysis),
sidebar (language, sessions, retrieval, Agentic RAG), and wiring to LLM, vector search,
web augmentation, and persisted chats.
"""

import streamlit as st

from src.config import load_llm_client
from src.loaders import load_document_chunks
from src.paper_summary import summarize_paper, chat_with_paper
from src.vector_store import retrieve
from src.llm import generate_chat_answer, generate_rag_answer
from src.agentic_rag import run_agentic_rag
from src.embeddings import query_embedder
from src.paper_structure import parse_paper, render_evidence_image, select_visual_evidence
from src.citations import evidence_records, validate_citations
from src.media import attachment_path
from src.ui import (sidebar_upgrade_settings, ensure_index, render_message, image_upload,
                    render_evidence, show_error, retrieval_feedback, render_usage, evaluation_panel,
                    index_for_chunks)
from src.api import text_model
from src.vector_store import corpus_fingerprint
from src.state import init_session_state, clear_knowledge_base
from src.chat import (
    init_chat_state,
    create_new_chat,
    get_current_chat,
    get_current_chat_id,
    get_current_messages,
    set_current_chat,
    add_message,
    get_recent_history,
    search_chats,
    get_recent_chats,
    clear_current_chat_messages,
    append_web_memory_record,
    get_current_web_memory,
    clear_current_web_memory,
    delete_chat,
)
from src.general_chat_web import refine_search_query, build_supplementary_context
from src.web_search import fetch_web_snippets
from src.i18n import (
    LANGUAGE_OPTIONS,
    init_language,
    set_language,
    get_language,
    t,
)

from src.paper_reader import (
    save_uploaded_pdf,
)
from src.data_analysis import (
    load_uploaded_table,
    dataframe_context_for_llm,
    generate_overview,
    generate_quality_issues,
    generate_analysis_report,
    auto_clean_dataframe,
    build_statistical_profile,
    correlation_heatmap_figure,
    generate_cleaning_explanation,
    generate_insight_summary,
    build_export_python_script,
)
from src.data_analysis_ext import (
    answer_data_question_smart,
    analyze_text_column,
    build_dashboard_figures,
    build_data_dictionary_chunks,
    build_dict_index,
    explain_training_result,
    importance_bar_figure,
    join_dataframes,
    train_simple_model,
)

try:
    import sklearn  # noqa: F401

    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False


# =========================
# 页面设置
# =========================
init_language()
st.set_page_config(page_title=t("page_title"), page_icon="📚", layout="wide")
st.title(t("app_title"))
st.markdown(t("app_caption"))


# =========================
# 初始化
# =========================
init_session_state()
init_chat_state()

client = load_llm_client()

if get_current_chat_id() is None:
    create_new_chat()


# =========================
# 侧边栏：模式 + 会话管理 + 设置
# =========================
with st.sidebar:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            letter-spacing: 0.02em;
        }
        .sidebar-brand {
            font-size: 1.05rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.35rem;
        }
        .sidebar-section-hint {
            font-size: 0.78rem;
            color: #64748b;
            margin-top: -0.4rem;
            margin-bottom: 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sidebar-brand">{t("sidebar_brand")}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="sidebar-section-hint">{t("sidebar_brand_hint")}</p>',
        unsafe_allow_html=True,
    )

    lang_codes = [c for c, _ in LANGUAGE_OPTIONS]
    lang_map = dict(LANGUAGE_OPTIONS)
    _cur = get_language()
    if _cur not in lang_codes:
        _cur = "zh"
    _lang_sel = st.selectbox(
        t("language_ui"),
        options=lang_codes,
        index=lang_codes.index(_cur),
        format_func=lambda c: lang_map[c],
        key="ui_language_select",
    )
    if _lang_sel != get_language():
        set_language(_lang_sel)
        st.rerun()

    st.subheader(t("section_mode"))
    _mode_labels = {"doc": "mode_doc", "paper": "mode_paper", "data": "mode_data", "eval": "mode_eval"}
    _mode_display = {code: t(key) for code, key in _mode_labels.items()}

    def _fmt_mode(m: str) -> str:
        return _mode_display[m]

    app_mode = st.radio(
        "mode_internal",
        ["doc", "paper", "data", "eval"],
        index=0,
        format_func=_fmt_mode,
        label_visibility="collapsed",
    )
    if app_mode == "doc":
        mode_hint = t("mode_hint_doc")
    elif app_mode == "paper":
        mode_hint = t("mode_hint_paper")
    elif app_mode == "data":
        mode_hint = t("mode_hint_data")
    else:
        mode_hint = t("eval_caption")
    st.caption(mode_hint)

    st.divider()

    st.subheader(t("section_session"))
    if st.button(t("new_chat"), use_container_width=True, type="primary"):
        create_new_chat()

    search_keyword = st.text_input(
        t("search_sessions"),
        placeholder=t("search_placeholder"),
    )

    st.caption(t("caption_recent_chats"))

    if search_keyword.strip():
        chat_list = search_chats(search_keyword)
    else:
        chat_list = get_recent_chats(limit=12)

    if not chat_list:
        st.info(t("no_sessions"), icon="💭")
    else:
        for chat in chat_list:
            cid = chat["id"]
            is_current = cid == get_current_chat_id()
            title = (chat.get("title") or t("untitled")).strip() or t("untitled")
            short_title = title if len(title) <= 22 else title[:19] + "…"
            updated = chat.get("updated_at", "")

            with st.container(border=True):
                row_main, row_more = st.columns([5, 1])
                with row_main:
                    btn_label = f"◆ {short_title}" if is_current else short_title
                    btn_kw = {"key": f"chat_btn_{cid}", "use_container_width": True}
                    if is_current:
                        btn_kw["type"] = "primary"
                    if st.button(btn_label, **btn_kw):
                        set_current_chat(cid)
                with row_more:
                    with st.popover("⋮", help=t("popover_more")):
                        st.markdown(f"**{title}**")
                        if updated:
                            st.caption(f"{t('updated_prefix')} {updated}")
                        if st.button(
                            t("btn_delete_chat"),
                            key=f"delete_chat_{cid}",
                            use_container_width=True,
                        ):
                            delete_chat(cid)
                            st.rerun()

    st.divider()
    st.subheader(t("section_kb"))

    top_k = st.slider(t("slider_top_k"), min_value=2, max_value=8, value=4, step=1)
    chunk_size = st.slider(t("slider_chunk_size"), min_value=300, max_value=1500, value=800, step=100)
    overlap = st.slider(t("slider_overlap"), min_value=50, max_value=min(400, chunk_size - 50), value=150, step=50)
    max_turns = st.slider(t("slider_max_turns"), min_value=1, max_value=10, value=3, step=1)

    st.subheader(t("section_agentic"))
    use_agentic_rag = st.checkbox(
        t("agentic_checkbox"),
        value=True,
        help=t("agentic_help"),
    )

    st.subheader(t("section_general_web"))
    general_chat_web = st.checkbox(
        t("general_web_checkbox"),
        value=True,
        help=t("general_web_help"),
    )
    max_general_web = st.slider(t("slider_general_web_results"), min_value=2, max_value=10, value=5, step=1)

    if st.button(t("btn_clear_web_memory"), use_container_width=True):
        clear_current_web_memory()
        st.success(t("msg_web_memory_cleared"))

    if st.button(t("btn_clear_kb"), use_container_width=True):
        clear_knowledge_base()
        st.success(t("msg_kb_cleared"))

    if st.button(t("btn_clear_messages"), use_container_width=True):
        clear_current_chat_messages()
        st.success(t("msg_chat_cleared"))


    embed_model, retrieval_config, agent_limits, pdf_parser, decompose_queries, agent_web = sidebar_upgrade_settings()
    usage_slot = st.empty()


# =========================
# 模式 1：普通/文档聊天
# =========================
if app_mode == "doc":
    st.markdown(t("doc_kb_heading"))

    uploaded_files = st.file_uploader(
        t("upload_files"),
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button(t("build_kb"), disabled=not uploaded_files):
            all_chunk_dicts = []

            with st.spinner(t("building_kb")):
                for file in uploaded_files:
                    try:
                        all_chunk_dicts.extend(load_document_chunks(file, chunk_size=chunk_size,
                                                                   overlap=overlap, parser=pdf_parser))
                    except Exception as e:
                        st.error(f"{file.name}: {show_error(e)}")

                if not all_chunk_dicts:
                    st.error(t("err_no_kb_text"))
                else:
                    try:
                        index = index_for_chunks(st.session_state.get('index'),all_chunk_dicts,embed_model)
                        st.session_state["chunks"] = all_chunk_dicts
                        st.session_state["index"] = index
                        st.session_state["kb_ready"] = True
                        st.success(t("success_kb_built", n=len(all_chunk_dicts)))
                    except Exception as e:
                        st.error(show_error(e))

    with col2:
        if st.session_state["kb_ready"]:
            st.info(t("info_kb_ready", n=len(st.session_state['chunks'])))
        else:
            st.warning(t("warn_kb_not_ready"))

    current_chat = get_current_chat()
    if current_chat:
        st.markdown(t("current_chat_heading", title=current_chat['title']))
        st.caption(f"{t('last_updated')}: {current_chat['updated_at']}")

    kb_ready = (
        st.session_state["kb_ready"]
        and st.session_state["index"] is not None
        and len(st.session_state["chunks"]) > 0
    )

    if kb_ready:
        if use_agentic_rag:
            st.success(t("status_agentic_on"))
        else:
            st.success(t("status_simple_rag"))
    else:
        if general_chat_web:
            st.info(t("status_general_web"))
        else:
            st.info(t("status_general_only"))

    for msg in get_current_messages():
        with st.chat_message(msg["role"]):
            render_message(msg)

    images, send_images, image_error, image_epoch_key = image_upload(get_current_chat_id())
    query = st.chat_input(t("chat_input_doc"))
    if send_images:
        query = query or t('image_default_question')
    if image_error:
        query = None

    if query:
        st.session_state[image_epoch_key] = st.session_state.get(image_epoch_key, 0) + 1
        add_message("user", query, attachments=images)
        retrieved_docs = []

        with st.chat_message("user"):
            render_message({"content": query, "attachments": images})

        try:
            with st.chat_message("assistant"):
                with st.spinner(t("generating")):
                    history = get_recent_history(max_turns=max_turns)
    
                    if kb_ready:
                        embed_fn = query_embedder(embed_model)
                        ensure_index("index", "chunks", embed_model)
    
                        if use_agentic_rag:
                            result = run_agentic_rag(
                                query=query,
                                history=history,
                                index=st.session_state["index"],
                                chunks=st.session_state["chunks"],
                                top_k=top_k,
                                embed_fn=embed_fn,
                                client=client,
                                router_domain="library",
                                lang_code=get_language(), retrieval_config=retrieval_config,
                                limits=agent_limits, allow_web=agent_web, decompose=decompose_queries,
                                images=images or [a for m in history for a in m.get('attachments', [])][-4:],
                            )
                            answer = result["answer"]
                            retrieved_docs = result["retrieved_docs"]
                            trace = result["trace"]
                        else:
                            retrieved_docs = retrieve(
                                query=query,
                                index=st.session_state["index"],
                                chunks=st.session_state["chunks"],
                                k=top_k,
                                embed_fn=embed_fn, config=retrieval_config,
                            )
                            answer = generate_rag_answer(
                                query=query,
                                retrieved_docs=retrieved_docs,
                                history=history,
                                client=client,
                                lang_code=get_language(), images=images or
                                    [a for m in history for a in m.get('attachments', [])][-4:],
                            )
                            answer, _ = validate_citations(answer, retrieved_docs)
                            trace = None
    
                        st.write(answer)
                        retrieval_feedback(retrieved_docs)
                        render_evidence(evidence_records(retrieved_docs))
    
                        if trace is not None:
                            with st.expander(t("expander_agentic_trace")):
                                st.caption(t('trace_caption'))
                                st.json(trace)
    
                        with st.expander(t("expander_context_chunks")):
                            for i, doc in enumerate(retrieved_docs, start=1):
                                score = doc.get("score")
                                score_s = f"{float(score):.4f}" if score is not None else "n/a"
                                st.markdown(
                                    t(
                                        "chunk_line",
                                        i=i,
                                        src=doc["source"],
                                        score=score_s,
                                    )
                                )
                                st.write(doc["text"])
                                st.divider()
                    else:
                        if general_chat_web and not images and not any(m.get('attachments') for m in history):
                            with st.spinner(t("spinner_web_memory")):
                                sq = refine_search_query(query, history, client)
                                raw = fetch_web_snippets(sq, max_results=max_general_web)
                                items = [
                                        {"source": x.get("source", ""), "text": x.get("text", ""),
                                         "url": x.get("url")}
                                    for x in raw
                                ]
                                append_web_memory_record(sq, items)
                                supplementary = build_supplementary_context(
                                    get_current_web_memory()
                                )
                                answer = generate_chat_answer(
                                    query=query,
                                    history=history,
                                    client=client,
                                    supplementary_context=supplementary,
                                    lang_code=get_language(),
                                )
                            st.write(answer)
                            st.caption(t("caption_web_chat"))
                            with st.expander(t("expander_web_round")):
                                st.markdown(t("web_query_line", q=sq, n=len(items)))
                                for i, it in enumerate(items, start=1):
                                    st.markdown(f"**{i}.** `{it['source']}`")
                                    st.write(it["text"][:1200])
                                    st.divider()
                            mem = get_current_web_memory()
                            st.caption(t("caption_memory_count", n=len(mem)))
                        else:
                            answer = generate_chat_answer(
                                query=query,
                                history=history,
                                client=client,
                                lang_code=get_language(), images=images,
                            )
                            st.write(answer)
                            st.caption(t("caption_plain_chat"))
        except Exception as e:
            answer = show_error(e)
            st.error(answer)

        add_message("assistant", answer, evidence=evidence_records(retrieved_docs))


# =========================
# 模式 2：论文阅读
# =========================
elif app_mode == "paper":
    st.markdown(t("paper_heading"))
    st.markdown(t("paper_caption"))

    uploaded_papers = st.file_uploader(
        t("upload_paper_pdf"),
        type=["pdf"],
        key="paper_pdf_uploader", accept_multiple_files=True
    )

    main_col, thumb_col = st.columns([2.4, 1])

    with main_col:
        if st.button(t("btn_parse_paper"), disabled=not uploaded_papers, use_container_width=True):
            with st.spinner(t("spinner_parsing")):
                try:
                    parsed_papers, paper_chunk_dicts, paper_images = [], [], []
                    for uploaded_paper in uploaded_papers:
                        pdf_path = save_uploaded_pdf(uploaded_paper)
                        parsed = parse_paper(pdf_path, uploaded_paper.name, parser=pdf_parser,
                                             chunk_size=chunk_size, overlap=overlap)
                        parsed_papers.append(parsed)
                        paper_chunk_dicts.extend(parsed['chunks'])
                        for visual in parsed['visuals'][:max(0, 12-len(paper_images))]:
                            image = render_evidence_image(visual)
                            paper_images.append({**visual, 'path': str(attachment_path(image)),
                                                 'name': image['name']})
                        st.caption(t('parser_used', parser=parsed['parser'], chunks=len(parsed['chunks']),
                                     visuals=len(parsed['visuals'])))
                        for warning in parsed['warnings']:
                            st.warning(t(warning))
                    paper_text = '\n\n'.join(p['text'] for p in parsed_papers)
                    new_paper_index = index_for_chunks(st.session_state.get('paper_index'),paper_chunk_dicts,embed_model)
                    summary_key = (corpus_fingerprint(paper_chunk_dicts),get_language(),text_model())
                    reuse_summary = st.session_state.get('_paper_summary_key') == summary_key
                    st.session_state['paper_text'] = paper_text
                    st.session_state['paper_images'] = paper_images
                    st.session_state['paper_chunks'] = paper_chunk_dicts
                    st.session_state['paper_ready'] = bool(paper_chunk_dicts)
                    st.session_state['paper_index'] = new_paper_index
                    st.session_state['paper_vector_ready'] = True
                    st.session_state['selected_paper_image'] = paper_images[0] if paper_images else None
                    try:
                        if not reuse_summary:
                            st.session_state['paper_summary'] = ''
                        if paper_text.strip() and not reuse_summary:
                            st.session_state['paper_summary'] = summarize_paper(paper_text, client, lang_code=get_language())
                            st.session_state['_paper_summary_key'] = summary_key
                    except Exception as e:
                        st.warning(show_error(e))

                    st.success(t("success_parsed"))
                    if st.session_state.get("paper_vector_ready"):
                        st.info(
                            t(
                                "info_vector_index",
                                n=len(st.session_state["paper_chunks"]),
                            )
                        )
                except Exception as e:
                    st.error(f"{t('err_parse')}: {show_error(e)}")

        st.markdown(t("section_selected_image"))

        selected_image = st.session_state.get("selected_paper_image")
        if selected_image:
            st.image(
                selected_image["path"],
                caption=t(
                    "image_caption",
                    page=selected_image["page"],
                    name=selected_image["name"],
                ),
                use_container_width=True
            )
        else:
            st.info(t("info_select_image_hint"))

        st.markdown("---")
        st.markdown(t("section_summary"))

        if st.session_state.get("paper_summary"):
            st.write(st.session_state["paper_summary"])

            with st.expander(t("expander_first_5k")):
                st.text(st.session_state.get("paper_text", "")[:5000])
        else:
            st.info(t("info_upload_first"))

        st.markdown("---")
        st.markdown(t("section_paper_chat"))

        paper_vector_ready = (
            bool(st.session_state.get("paper_vector_ready"))
            and st.session_state.get("paper_index") is not None
            and len(st.session_state.get("paper_chunks") or []) > 0
        )
        if paper_vector_ready:
            if use_agentic_rag:
                st.success(t("status_paper_agentic_on"))
            else:
                st.warning(t("status_paper_simple_rag"))
        else:
            st.info(t("status_paper_fallback"))

        current_chat = get_current_chat()
        if current_chat:
            st.caption(t("current_chat_caption", title=current_chat["title"]))

        for msg in get_current_messages():
            with st.chat_message(msg["role"]):
                render_message(msg)

        force_paper_visual = st.checkbox(t('force_visual'), value=False)
        paper_query = st.chat_input(t("paper_chat_input"), key="paper_chat_input")

        if paper_query:
            retrieved_docs = []
            add_message("user", paper_query)

            with st.chat_message("user"):
                st.write(paper_query)

            try:
                with st.chat_message("assistant"):
                    with st.spinner(t("spinner_paper_answer")):
                        history = get_recent_history(max_turns=max_turns)
    
                        paper_ready = (
                            st.session_state.get("paper_ready", False)
                            and bool(st.session_state.get("paper_chunks") or st.session_state.get("paper_text", "").strip())
                        )
                        paper_vector_ready = (
                            bool(st.session_state.get("paper_vector_ready"))
                            and st.session_state.get("paper_index") is not None
                            and len(st.session_state.get("paper_chunks") or []) > 0
                        )
    
                        if paper_ready and paper_vector_ready:
                            embed_fn = query_embedder(embed_model)
                            ensure_index("paper_index", "paper_chunks", embed_model)
                            if use_agentic_rag:
                                result = run_agentic_rag(
                                    query=paper_query,
                                    history=history,
                                    index=st.session_state["paper_index"],
                                    chunks=st.session_state["paper_chunks"],
                                    top_k=top_k,
                                    embed_fn=embed_fn,
                                    client=client,
                                    router_domain="paper",
                                    lang_code=get_language(), retrieval_config=retrieval_config,
                                    limits=agent_limits, allow_web=agent_web, decompose=decompose_queries,
                                    force_visual=force_paper_visual,
                                )
                                answer = result["answer"]
                                retrieved_docs = result["retrieved_docs"]
                                trace = result["trace"]
                            else:
                                retrieved_docs = retrieve(
                                    query=paper_query,
                                    index=st.session_state["paper_index"],
                                    chunks=st.session_state["paper_chunks"],
                                    k=top_k,
                                    embed_fn=embed_fn, config=retrieval_config,
                                )
                                visual_query = paper_query + '\n' + '\n'.join(
                                    m['content'] for m in history[-4:] if m.get('role') == 'user')
                                selected, paper_visuals = select_visual_evidence(
                                    visual_query, retrieved_docs, st.session_state['paper_chunks'],
                                    agent_limits.max_visuals, force=force_paper_visual)
                                known = {d.get('chunk_id') for d in retrieved_docs}
                                retrieved_docs += [d for d in selected if d.get('chunk_id') not in known]
                                answer = generate_rag_answer(
                                    query=paper_query,
                                    retrieved_docs=retrieved_docs,
                                    history=history,
                                    client=client,
                                    lang_code=get_language(), images=paper_visuals,
                                )
                                answer, _ = validate_citations(answer, retrieved_docs)
                                trace = None
    
                            st.write(answer)
                            retrieval_feedback(retrieved_docs)
                            render_evidence(evidence_records(retrieved_docs))
                            st.caption(t("caption_paper_rag"))
                            if trace is not None:
                                with st.expander(t("expander_paper_trace")):
                                    st.caption(t('trace_caption'))
                                    st.json(trace)
                            with st.expander(t("expander_paper_chunks")):
                                for i, doc in enumerate(retrieved_docs, start=1):
                                    score = doc.get("score")
                                    score_s = (
                                        f"{float(score):.4f}"
                                        if score is not None
                                        else "n/a"
                                    )
                                    st.markdown(
                                        t(
                                            "chunk_line",
                                            i=i,
                                            src=doc["source"],
                                            score=score_s,
                                        )
                                    )
                                    st.write(doc["text"])
                                    st.divider()
                        elif paper_ready:
                            answer = chat_with_paper(
                                paper_text=st.session_state["paper_text"],
                                query=paper_query,
                                history=history,
                                client=client,
                                lang_code=get_language(),
                            )
                            st.write(answer)
                            st.caption(t("caption_full_paper_mode"))
                        else:
                            answer = t("warn_upload_first")
                            st.warning(answer)
            except Exception as e:
                answer = show_error(e)
                st.error(answer)

            add_message("assistant", answer, evidence=evidence_records(retrieved_docs))

    with thumb_col:
        st.markdown(t("section_thumbnails"))

        paper_images = st.session_state.get("paper_images", [])

        if not paper_images:
            st.info(t("info_thumbnails_empty"))
        else:
            thumb_container = st.container(height=1000)

            with thumb_container:
                for i, img in enumerate(paper_images, start=1):
                    st.image(
                        img["path"],
                        caption=t("thumb_caption", i=i, page=img["page"]),
                        use_container_width=True
                    )

                    if st.button(
                        t("btn_view_figure", i=i),
                        key=f"select_paper_img_{i}",
                        use_container_width=True
                    ):
                        st.session_state["selected_paper_image"] = img
                        st.rerun()

                    st.divider()


# =========================
# 模式 3：数据分析
# =========================
elif app_mode == "data":
    st.markdown(t("data_heading"))
    st.markdown(t("data_caption"))

    da_file = st.file_uploader(
        t("data_upload"),
        type=["csv", "xlsx", "xls"],
        key="da_file_uploader",
    )

    b1, b2 = st.columns([2, 1])
    with b1:
        load_clicked = st.button(
            t("data_btn_apply"),
            disabled=da_file is None,
            use_container_width=True,
            type="primary",
            key="da_btn_load",
        )
    with b2:
        if st.button(t("data_btn_clear"), use_container_width=True, key="da_btn_clear"):
            st.session_state["da_df"] = None
            st.session_state["da_ready"] = False
            st.session_state["da_filename"] = ""
            st.session_state["da_overview"] = ""
            st.session_state["da_quality"] = ""
            st.session_state["da_context"] = ""
            st.session_state["da_last_report"] = ""
            st.session_state["da_clean_steps"] = []
            st.session_state["da_clean_explain"] = ""
            st.session_state["da_insights"] = ""
            st.session_state["da_stats_text"] = ""
            st.session_state["da_before_rows"] = 0
            st.session_state["da_side_tables"] = {}
            st.session_state["da_dict_index"] = None
            st.session_state["da_dict_chunks"] = []
            st.success(t("data_cleared"))
            st.rerun()

    if load_clicked and da_file is not None:
        with st.spinner(t("data_spinner_load")):
            try:
                df_raw = load_uploaded_table(da_file)
                before_n = len(df_raw)
                df0, clean_steps = auto_clean_dataframe(df_raw)
                st.session_state['da_dict_index'] = None
                st.session_state['da_dict_chunks'] = []
                st.session_state['da_last_report'] = ''
                st.session_state['da_ml_last'] = None
                st.session_state["da_before_rows"] = before_n
                st.session_state["da_df"] = df0
                st.session_state["da_clean_steps"] = clean_steps
                st.session_state["da_filename"] = da_file.name or "data"
                st.session_state["da_ready"] = True
                ctx = dataframe_context_for_llm(df0)
                st.session_state["da_context"] = ctx
                st.session_state["da_overview"] = generate_overview(
                    client, ctx, get_language()
                )
                st.session_state["da_quality"] = generate_quality_issues(
                    client, ctx, get_language()
                )
                prof = build_statistical_profile(df0)
                st.session_state["da_stats_text"] = prof["text_for_llm"]
                st.session_state["da_insights"] = generate_insight_summary(
                    client,
                    ctx,
                    prof["text_for_llm"],
                    get_language(),
                )
                st.session_state["da_clean_explain"] = generate_cleaning_explanation(
                    client,
                    clean_steps,
                    before_n,
                    len(df0),
                    get_language(),
                )
                st.success(
                    t(
                        "data_success",
                        name=st.session_state["da_filename"],
                        rows=len(df0),
                        cols=len(df0.columns),
                    )
                )
            except Exception as e:
                st.error(f"{t('data_err')}: {e}")
                st.session_state["da_ready"] = False

    if st.session_state.get("da_ready") and st.session_state.get("da_df") is not None:
        df = st.session_state["da_df"]

        def _da_step_line(step: dict) -> str:
            k = step.get("key")
            if not k:
                return str(step)
            kw = {a: b for a, b in step.items() if a != "key"}
            return t(f"data_log_{k}", **kw)

        st.markdown(t("data_section_clean"))
        st.caption(
            f"{st.session_state.get('da_before_rows', 0)} → {len(df)} rows"
        )
        for s in st.session_state.get("da_clean_steps") or []:
            st.markdown(f"- {_da_step_line(s)}")
        st.markdown(t("data_clean_ai"))
        st.markdown(st.session_state.get("da_clean_explain") or "")

        st.markdown(t("data_section_overview"))
        st.markdown(st.session_state.get("da_overview") or "")

        st.markdown(t("data_section_quality"))
        st.markdown(st.session_state.get("da_quality") or "")

        st.markdown(t("data_section_insights"))
        st.markdown(st.session_state.get("da_insights") or "")

        st.markdown(t("data_section_stats"))
        with st.expander(t("data_stats_expander"), expanded=False):
            st.text(st.session_state.get("da_stats_text", "")[:12000])
        num_df = df.select_dtypes(include="number")
        if len(num_df.columns) > 0:
            st.caption(t("data_stats_skew"))
            st.dataframe(
                num_df.skew(numeric_only=True).to_frame("skew"),
                use_container_width=True,
            )
        ch = correlation_heatmap_figure(df)
        if ch is not None:
            st.caption(t("data_stats_corr_caption"))
            st.plotly_chart(ch, use_container_width=True, key='da_correlation')

        st.markdown(t("data_section_preview"))
        st.dataframe(df.head(50), use_container_width=True)

        tab_join, tab_dash, tab_ml, tab_txt, tab_dic = st.tabs(
            [
                t("data_tab_join"),
                t("data_tab_dash"),
                t("data_tab_ml"),
                t("data_tab_text"),
                t("data_tab_dict"),
            ]
        )

        primary_label = st.session_state.get("da_filename") or "primary"

        with tab_join:
            st.caption(t("data_join_help"))
            extra_files = st.file_uploader(
                t("data_join_extra"),
                type=["csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="da_join_extra_uploader",
            )
            if extra_files:
                for f in extra_files:
                    try:
                        raw_e = load_uploaded_table(f)
                        df_e, _ = auto_clean_dataframe(raw_e)
                        st.session_state["da_side_tables"][f.name or "extra"] = df_e
                    except Exception as ex:
                        st.warning(f"{f.name}: {ex}")

            side = st.session_state.get("da_side_tables") or {}
            lab_map = {"__main__": primary_label}
            for k in side:
                lab_map[k] = k
            opt_ids = ["__main__"] + sorted(side.keys())

            if not side:
                st.info(t("data_join_need_extra"))
            else:
                cj1, cj2 = st.columns(2)
                with cj1:
                    left_id = st.selectbox(
                        t("data_join_left"),
                        opt_ids,
                        format_func=lambda x: lab_map.get(x, x),
                        key="da_join_left_id",
                    )
                right_candidates = [x for x in opt_ids if x != left_id]
                with cj2:
                    right_id = st.selectbox(
                        t("data_join_right"),
                        right_candidates,
                        format_func=lambda x: lab_map.get(x, x),
                        key="da_join_right_id",
                    )

                how = st.selectbox(
                    t("data_join_how"),
                    ["inner", "left", "right", "outer"],
                    key="da_join_how",
                )

                left_df = df if left_id == "__main__" else side.get(left_id)
                right_df = df if right_id == "__main__" else side.get(right_id)
                lcols = left_df.columns.tolist()
                rcols = right_df.columns.tolist()
                cj3, cj4 = st.columns(2)
                with cj3:
                    lk = st.selectbox(t("data_join_left_on"), lcols, key="da_join_lk")
                with cj4:
                    rk = st.selectbox(t("data_join_right_on"), rcols, key="da_join_rk")
                if st.button(t("data_join_btn"), key="da_join_run"):
                    try:
                        merged = join_dataframes(
                            left_df, right_df, how, lk, rk
                        )
                        st.session_state["da_df"] = merged
                        st.session_state["da_context"] = dataframe_context_for_llm(merged)
                        st.session_state['da_dict_index'] = None
                        st.session_state['da_dict_chunks'] = []
                        st.session_state['da_last_report'] = ''
                        st.session_state['da_ml_last'] = None
                        st.session_state['da_stats_text'] = build_statistical_profile(merged)['text_for_llm']
                        for _derived in ('da_overview','da_quality','da_insights'):
                            st.session_state[_derived] = ''
                        st.success(
                            t(
                                "data_join_ok",
                                rows=len(merged),
                                cols=len(merged.columns),
                            )
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"{t('data_join_err')}: {e}")

        with tab_dash:
            st.markdown(f"### {t('data_dash_title')}")
            num_top = df.select_dtypes(include="number").columns.tolist()[:4]
            if num_top:
                r1 = st.columns(len(num_top))
                for i, c in enumerate(num_top):
                    with r1[i]:
                        st.metric(label=str(c), value=f"{df[c].mean():.4g}")
            for chart_i, (title, fig) in enumerate(build_dashboard_figures(df)):
                st.caption(title)
                st.plotly_chart(fig, use_container_width=True, key=f'da_dashboard_{chart_i}')

        with tab_ml:
            st.markdown(f"### {t('data_ml_title')}")
            if not _HAVE_SKLEARN:
                st.warning(t("data_ml_need_sklearn"))
            tgt = st.selectbox(
                t("data_ml_target"),
                df.columns.tolist(),
                key="da_ml_target",
            )
            if st.button(t("data_ml_run"), key="da_ml_btn"):
                with st.spinner(t("data_spinner_answer")):
                    tr = train_simple_model(df, tgt)
                    if tr.get("ok"):
                        st.session_state["da_ml_last"] = tr
                        expl = explain_training_result(client, tr, get_language())
                        st.markdown(expl)
                        figi = importance_bar_figure(
                            tr.get("importance") or [],
                            title=t("data_ml_title"),
                        )
                        if figi is not None:
                            st.plotly_chart(figi, use_container_width=True, key='da_model_importance')
                        st.json(tr.get("metrics") or {})
                    else:
                        st.error(tr.get("error", "error"))

        with tab_txt:
            st.markdown(f"### {t('data_text_title')}")
            text_cols = [
                c
                for c in df.columns
                if df[c].dtype == object
                or str(df[c].dtype).startswith(("string", "category"))
            ]
            if not text_cols:
                st.caption("—")
            else:
                tc = st.selectbox(
                    t("data_text_col"),
                    text_cols,
                    key="da_text_col",
                )
                if st.button(t("data_text_run"), key="da_text_btn"):
                    with st.spinner(t("data_spinner_answer")):
                        prof = analyze_text_column(df[tc], client, get_language())
                        st.markdown(prof.get("summary") or "")
                        terms = prof.get("top_terms") or []
                        if terms:
                            st.caption(
                                ", ".join(f"{a} ({b:.0f})" for a, b in terms[:12])
                            )

        with tab_dic:
            st.markdown(f"### {t('data_dict_title')}")
            if st.button(t("data_dict_build"), key="da_dict_build"):
                with st.spinner(t("data_spinner_answer")):
                    chunks = build_data_dictionary_chunks(df)
                    idx, ch = build_dict_index(chunks, embed_model)
                    st.session_state["da_dict_index"] = idx
                    st.session_state["da_dict_chunks"] = ch
                    st.success(t("data_dict_built", n=len(ch)))
            if st.session_state.get("da_dict_index") is not None:
                st.caption(t("data_dict_chat_hint"))

        current_chat = get_current_chat()
        if current_chat:
            st.caption(t("current_chat_caption", title=current_chat["title"]))

        for msg in get_current_messages():
            with st.chat_message(msg["role"]):
                render_message(msg)

        da_q = st.chat_input(
            t("data_chat_placeholder"),
            key="da_chat_input",
        )

        if da_q:
            add_message("user", da_q)
            with st.chat_message("user"):
                st.write(da_q)
            with st.chat_message("assistant"):
                try:
                    with st.spinner(t("data_spinner_answer")):
                        hist = get_recent_history(max_turns=max_turns)
                        ctx_full = st.session_state.get("da_context") or dataframe_context_for_llm(
                            df
                        )
                        if st.session_state.get('da_dict_chunks'):
                            ensure_index('da_dict_index', 'da_dict_chunks', embed_model)
                        out = answer_data_question_smart(
                            client,
                            df,
                            ctx_full,
                            da_q,
                            hist,
                            get_language(),
                            dict_index=st.session_state.get("da_dict_index"),
                            dict_chunks=st.session_state.get("da_dict_chunks"),
                            embed_model=embed_model,
                        )
                    rn = out.get("route_note") or ""
                    intent = out.get("intent") or ""
                    if intent or rn:
                        st.caption(
                            f"{t('data_route_label')}: `{intent}` — {rn}"
                        )
                    st.markdown(out.get("explanation") or "")
                    tbl = out.get("table_df")
                    if tbl is not None and not tbl.empty:
                        with st.expander(t("data_expander_table"), expanded=True):
                            st.dataframe(tbl, use_container_width=True)
                    fig = out.get("figure")
                    if fig is not None:
                        with st.expander(t("data_expander_chart"), expanded=True):
                            st.plotly_chart(fig, use_container_width=True, key='da_question_chart')
                    with st.expander(t("data_expander_debug")):
                        st.json(out.get("raw_json") or {})
                    _ans_text = out.get("explanation") or ""
                except Exception as e:
                    _ans_text = f"{t('data_err')}: {e}"
                    st.error(_ans_text)
                add_message("assistant", _ans_text)

        st.markdown("---")
        if st.button(t("data_btn_report"), use_container_width=True, key="da_report_btn"):
            with st.spinner(t("data_spinner_report")):
                ctx_rep = st.session_state.get("da_context") or dataframe_context_for_llm(
                    df
                )
                msgs = get_current_messages()
                chat_txt = "\n".join(
                    f"{m.get('role', '')}: {m.get('content', '')[:2000]}"
                    for m in msgs[-20:]
                )
                report_md = generate_analysis_report(
                    client,
                    st.session_state.get("da_overview") or "",
                    st.session_state.get("da_quality") or "",
                    ctx_rep,
                    chat_txt,
                    get_language(),
                    cleaning_explanation=st.session_state.get("da_clean_explain") or "",
                    insights=st.session_state.get("da_insights") or "",
                    stats_excerpt=st.session_state.get("da_stats_text") or "",
                )
                st.session_state["da_last_report"] = report_md

        if st.session_state.get("da_last_report"):
            st.markdown(t("data_report_title"))
            st.markdown(st.session_state["da_last_report"])
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    label=t("data_download_md"),
                    data=st.session_state["da_last_report"],
                    file_name="analysis_report.md",
                    mime="text/markdown",
                    key="da_download_md",
                )
            with c_dl2:
                _py = build_export_python_script(
                    st.session_state.get("da_filename") or "data.csv"
                )
                st.download_button(
                    label=t("data_btn_export_py"),
                    data=_py,
                    file_name="data_pipeline.py",
                    mime="text/x-python",
                    key="da_download_py",
                    help=t("data_export_py_help"),
                )
    else:
        st.info(t("data_need_file"))

elif app_mode == "eval":
    evaluation_panel(embed_model, retrieval_config)

with usage_slot.container():
    render_usage(client)
