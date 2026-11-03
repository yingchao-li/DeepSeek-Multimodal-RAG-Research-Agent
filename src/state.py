"""
Streamlit session_state initialization: knowledge-base chunks/index, paper PDF metadata and vector index,
and data-analysis DataFrame plus auxiliary join/dictionary-RAG keys; includes clear_kb helper.
"""

# import streamlit as st
#
#
# def init_session_state():
#     if "chunks" not in st.session_state:
#         st.session_state["chunks"] = []
#
#     if "index" not in st.session_state:
#         st.session_state["index"] = None
#
#     if "kb_ready" not in st.session_state:
#         st.session_state["kb_ready"] = False
#
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = []
#
#
# def clear_knowledge_base():
#     st.session_state["chunks"] = []
#     st.session_state["index"] = None
#     st.session_state["kb_ready"] = False
#
#
# def clear_all():
#     st.session_state["chunks"] = []
#     st.session_state["index"] = None
#     st.session_state["kb_ready"] = False
#     st.session_state["messages"] = []
import streamlit as st


def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state["chunks"] = []

    if "index" not in st.session_state:
        st.session_state["index"] = None

    if "kb_ready" not in st.session_state:
        st.session_state["kb_ready"] = False

    if "paper_pdf_path" not in st.session_state:
        st.session_state["paper_pdf_path"] = ""

    if "paper_text" not in st.session_state:
        st.session_state["paper_text"] = ""

    if "paper_images" not in st.session_state:
        st.session_state["paper_images"] = []

    if "paper_summary" not in st.session_state:
        st.session_state["paper_summary"] = ""

    if "paper_ready" not in st.session_state:
        st.session_state["paper_ready"] = False

    if "selected_paper_image" not in st.session_state:
        st.session_state["selected_paper_image"] = None

    if "paper_chunks" not in st.session_state:
        st.session_state["paper_chunks"] = []

    if "paper_index" not in st.session_state:
        st.session_state["paper_index"] = None

    if "paper_vector_ready" not in st.session_state:
        st.session_state["paper_vector_ready"] = False

    if "da_df" not in st.session_state:
        st.session_state["da_df"] = None
    if "da_ready" not in st.session_state:
        st.session_state["da_ready"] = False
    if "da_filename" not in st.session_state:
        st.session_state["da_filename"] = ""
    if "da_overview" not in st.session_state:
        st.session_state["da_overview"] = ""
    if "da_quality" not in st.session_state:
        st.session_state["da_quality"] = ""
    if "da_context" not in st.session_state:
        st.session_state["da_context"] = ""
    if "da_last_report" not in st.session_state:
        st.session_state["da_last_report"] = ""
    if "da_clean_steps" not in st.session_state:
        st.session_state["da_clean_steps"] = []
    if "da_clean_explain" not in st.session_state:
        st.session_state["da_clean_explain"] = ""
    if "da_insights" not in st.session_state:
        st.session_state["da_insights"] = ""
    if "da_stats_text" not in st.session_state:
        st.session_state["da_stats_text"] = ""
    if "da_before_rows" not in st.session_state:
        st.session_state["da_before_rows"] = 0
    if "da_side_tables" not in st.session_state:
        st.session_state["da_side_tables"] = {}
    if "da_dict_index" not in st.session_state:
        st.session_state["da_dict_index"] = None
    if "da_dict_chunks" not in st.session_state:
        st.session_state["da_dict_chunks"] = []

def clear_knowledge_base():
    st.session_state["chunks"] = []
    st.session_state["index"] = None
    st.session_state["kb_ready"] = False