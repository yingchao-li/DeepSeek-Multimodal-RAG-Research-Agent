"""
Extensions for data mode: multi-table joins, lightweight LLM intent routing, heuristic chart fallback,
sklearn baselines with explanations, text-column profiling, column-dictionary FAISS RAG, and auto dashboard figures.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data_analysis import (
    LANG_NAMES,
    _chat,
    _parse_json_object,
    answer_data_question,
    build_chart,
)
from src.vector_store import retrieve

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from sklearn.feature_extraction.text import CountVectorizer

    _TEXT_OK = True
except ImportError:
    _TEXT_OK = False


def join_dataframes(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str,
    left_on: str,
    right_on: str,
) -> pd.DataFrame:
    how = (how or "inner").lower()
    if how not in ("inner", "left", "right", "outer"):
        how = "inner"
    if left_on not in left.columns or right_on not in right.columns:
        raise ValueError("Join keys must exist in both tables")
    return pd.merge(left, right, how=how, left_on=left_on, right_on=right_on)


def route_data_intent(
    client,
    question: str,
    schema_cols: list[str],
    lang_code: str,
) -> dict[str, Any]:
    """轻量 Agent 路由：识别问答 / 建模 / 文本 / 字典 / Join 等意图。"""
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是意图分类器。用户用{lang}提问。只输出一个 JSON 对象，不要其它文字：\n"
        '{"intent": "qa|train|text|schema|join|chart", "note": "简短理由"}\n'
        "含义：qa=一般统计问答；train=预测/建模/回归/分类/机器学习；"
        "text=文本挖掘/词频/情感/评论分析；schema=字段含义/列说明/数据字典；"
        "join=多表合并/关联/连接；chart=只要图/可视化。"
    )
    user = f"列名列表: {schema_cols[:80]}\n\n问题:\n{question[:2000]}"
    raw = _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        0.0,
    )
    data = _parse_json_object(raw) or {}
    intent = (data.get("intent") or "qa").lower()
    if intent not in ("qa", "train", "text", "schema", "join", "chart"):
        intent = "qa"
    return {"intent": intent, "note": data.get("note") or "", "raw": raw}


def heuristic_chart_spec(df: pd.DataFrame, question: str) -> Optional[dict[str, Any]]:
    """无 LLM 的图表类型启发式，用于补全空图。"""
    if df is None or df.empty:
        return None
    q = (question or "").lower()
    num = df.select_dtypes(include="number")
    cats = [
        c
        for c in df.columns
        if df[c].dtype == object or str(df[c].dtype).startswith("category")
    ]
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]

    if any(k in q for k in ("散点", "相关", "scatter", "correlation")) and num.shape[1] >= 2:
        c0, c1 = num.columns[0], num.columns[1]
        return {"type": "scatter", "x": c0, "y": c1}

    if any(k in q for k in ("时间", "趋势", "line", "trend")) and dt_cols and len(num.columns) > 0:
        return {"type": "line", "x": dt_cols[0], "y": num.columns[0]}

    if any(k in q for k in ("分布", "直方", "histogram", "distribution")) and len(num.columns) > 0:
        return {"type": "histogram", "x": num.columns[0]}

    if any(k in q for k in ("箱线", "box", "异常")) and len(num.columns) > 0:
        xc = cats[0] if cats else None
        yc = num.columns[0]
        if xc:
            return {"type": "box", "x": xc, "y": yc}
        return {"type": "box", "y": yc}

    if len(cats) > 0 and len(num.columns) > 0:
        best_cat = min(cats, key=lambda c: df[c].nunique(dropna=True))
        if df[best_cat].nunique(dropna=True) <= 40:
            return {
                "type": "bar",
                "group_cols": [best_cat],
                "agg": {num.columns[0]: "mean"},
            }

    if len(num.columns) >= 2:
        return {"type": "scatter", "x": num.columns[0], "y": num.columns[1]}
    if len(num.columns) == 1:
        return {"type": "histogram", "x": num.columns[0]}
    if cats:
        return {"type": "bar", "x": cats[0], "y": None}
    return None


def extract_column_name(
    client,
    question: str,
    columns: list[str],
    purpose: str,
    lang_code: str,
) -> Optional[str]:
    """从问题中解析单列名（建模目标或文本列）。"""
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"只输出 JSON：{{\"column\": \"必须是下列之一或 null\"}}。用途：{purpose}。用户语言：{lang}。"
    )
    user = f"列名: {json.dumps(columns, ensure_ascii=False)}\n问题:\n{question[:1500]}"
    raw = _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        0.0,
    )
    data = _parse_json_object(raw) or {}
    col = data.get("column")
    if isinstance(col, str) and col in columns:
        return col
    return None


def train_simple_model(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """随机森林基线 + 特征重要性（树模型）。"""
    if not _SKLEARN_OK:
        return {"ok": False, "error": "sklearn_not_installed"}

    if target_col not in df.columns:
        return {"ok": False, "error": "missing_target"}

    y_raw = df[target_col]
    row_ok = y_raw.notna()
    nunique = y_raw[row_ok].nunique()
    is_classification = bool(
        y_raw.dtype == object
        or str(y_raw.dtype).startswith("category")
        or (pd.api.types.is_integer_dtype(y_raw) and nunique <= 20)
    )

    X = df.drop(columns=[target_col]).copy()
    X = X.loc[row_ok]
    y_raw = y_raw.loc[row_ok]
    for c in list(X.columns):
        if X[c].dtype == object or str(X[c].dtype).startswith("category"):
            if X[c].nunique(dropna=False) > 40:
                X = X.drop(columns=[c])

    if X.shape[1] == 0:
        return {"ok": False, "error": "no_features"}

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols)
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", max_categories=30),
                        ),
                    ]
                ),
                cat_cols,
            )
        )

    if not transformers:
        return {"ok": False, "error": "no_features"}

    pre = ColumnTransformer(transformers, remainder="drop")

    if is_classification:
        if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
            y, _ = pd.factorize(y_raw)
        else:
            y = y_raw.to_numpy()
        y = np.asarray(y)
        if len(np.unique(y)) < 2 or len(y) < 10:
            return {"ok": False, "error": "too_few_rows"}
        strat = y if len(np.unique(y)) > 1 else None
        model = RandomForestClassifier(
            n_estimators=80, max_depth=12, random_state=42, n_jobs=-1
        )
        counts = np.unique(y, return_counts=True)[1]
        if counts.min() < 2 or int(np.ceil(len(y)*0.25)) < len(counts):
            return {'ok': False, 'error': 'too_few_rows_per_class'}
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=strat)
        X_train = pre.fit_transform(X_train_raw)
        X_test = pre.transform(X_test_raw)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        }
        task = "classification"
    else:
        y = pd.to_numeric(y_raw, errors="coerce")
        ok = y.notna()
        X = X.loc[ok]
        y = y.loc[ok]
        if len(y) < 10:
            return {"ok": False, "error": "too_few_rows"}
        model = RandomForestRegressor(
            n_estimators=80, max_depth=12, random_state=42, n_jobs=-1
        )
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        X_train = pre.fit_transform(X_train_raw)
        X_test = pre.transform(X_test_raw)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
        }
        task = "regression"

    feat_names = pre.get_feature_names_out()
    imp = getattr(model, "feature_importances_", None)
    importance: list[tuple[str, float]] = []
    if imp is not None and len(feat_names) == len(imp):
        pairs = sorted(zip(feat_names, imp), key=lambda x: -x[1])[:20]
        importance = [(str(a), float(b)) for a, b in pairs]

    return {
        "ok": True,
        "task": task,
        "target": target_col,
        "metrics": metrics,
        "importance": importance,
        "n_rows": len(df),
    }


def explain_training_result(
    client,
    result: dict[str, Any],
    lang_code: str,
) -> str:
    if not result.get("ok"):
        return ""
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是数据科学家。根据下列模型结果，用{lang}写 4～8 句：任务类型、指标含义、"
        "哪些特征重要、业务上可如何解读；不要编造数据中未出现的列名。"
    )
    user = json.dumps(result, ensure_ascii=False, indent=2)[:8000]
    return _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        0.3,
    )


def importance_bar_figure(importance: list[tuple[str, float]], title: str = "Feature importance"):
    if not importance:
        return None
    names = [x[0][:60] for x in importance[:15]]
    vals = [x[1] for x in importance[:15]]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h"))
    fig.update_layout(title=title, margin=dict(l=120, r=20, t=40, b=40))
    return fig


def analyze_text_column(
    series: pd.Series,
    client,
    lang_code: str,
    max_docs: int = 500,
) -> dict[str, Any]:
    """文本列：长度统计 + 词频（英文）+ LLM 摘要。"""
    s = series.dropna().astype(str)
    s = s[s.str.len() > 0]
    out: dict[str, Any] = {
        "n": len(s),
        "mean_len": float(s.str.len().mean()) if len(s) else 0.0,
        "top_terms": [],
        "summary": "",
    }
    if len(s) == 0:
        return out

    sample = s.head(30).tolist()
    if _TEXT_OK and len(s) > 5:
        try:
            vec = CountVectorizer(max_features=25, stop_words="english", ngram_range=(1, 1))
            X = vec.fit_transform(s.head(max_docs))
            sums = np.asarray(X.sum(axis=0)).ravel()
            terms = vec.get_feature_names_out()
            idx = np.argsort(-sums)[:15]
            out["top_terms"] = [(str(terms[i]), float(sums[i])) for i in idx]
        except Exception:
            pass

    lang = LANG_NAMES.get(lang_code, "中文")
    blob = "\n".join(sample[:15])[:6000]
    system = (
        f"根据下列文本样本（可能含中文/英文），用{lang}写 3～6 句：主题、语气、"
        "潜在情感倾向、注意噪声；不要编造具体事实。"
    )
    out["summary"] = _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": blob}],
        0.35,
    )
    return out


def build_data_dictionary_chunks(df: pd.DataFrame) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for col in df.columns:
        null_pct = float(df[col].isna().mean()) if len(df) else 0.0
        nu = int(df[col].nunique(dropna=True))
        sample = df[col].dropna().head(5).tolist()
        text = (
            f"列名 `{col}`：dtype={df[col].dtype}，缺失比例≈{null_pct:.2%}，"
            f"唯一值数量≈{nu}，示例值：{sample}"
        )
        chunks.append({"text": text, "source": f"column:{col}"})
    return chunks


def build_dict_index(
    chunks: list[dict[str, Any]],
    embed_model,
) -> tuple[Any, list[dict[str, Any]]]:
    from src.vector_store import build_retrieval_index
    index = build_retrieval_index(chunks, embed_model)
    return index, chunks


def embed_query_fn(embed_model):
    from src.embeddings import query_embedder
    return query_embedder(embed_model)


def answer_schema_rag(
    client,
    question: str,
    index,
    chunks: list[dict[str, Any]],
    embed_model,
    history: list[dict],
    lang_code: str,
) -> str:
    fn = embed_query_fn(embed_model)
    hits = retrieve(question, index, chunks, min(8, len(chunks)), fn)
    if not hits:
        return ""
    lang = LANG_NAMES.get(lang_code, "中文")
    ctx = "\n\n".join(
        f"[{h.get('source', '')}]\n{h.get('text', '')}" for h in hits[:8]
    )
    hist_lines = []
    for m in history[-6:]:
        hist_lines.append(f"{m.get('role', '')}: {m.get('content', '')[:800]}")
    system = (
        f"你是数据字典助手。仅根据下列「字段说明」回答用户关于列含义、类型、缺失与示例的问题。"
        f"用{lang}回答，条理清晰；若材料不足请说明。"
    )
    user = f"--- 字段说明 ---\n{ctx[:12000]}\n\n--- 最近对话 ---\n" + "\n".join(
        hist_lines
    ) + f"\n\n当前问题:\n{question}"
    return _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        0.2,
    )


def build_dashboard_figures(df: pd.DataFrame, max_charts: int = 4) -> list[tuple[str, Any]]:
    """自动选图：指标卡 + 若干 Plotly 图。"""
    figs: list[tuple[str, Any]] = []
    num = df.select_dtypes(include="number")
    cats = [
        c
        for c in df.columns
        if (df[c].dtype == object or str(df[c].dtype).startswith("category"))
        and df[c].nunique(dropna=False) <= 50
    ]

    if len(num.columns) > 0:
        c0 = num.columns[0]
        figs.append((f"Histogram: {c0}", px.histogram(df, x=c0)))

    if len(cats) > 0 and len(num.columns) > 0:
        cc, nc = cats[0], num.columns[0]
        sub = df.groupby(cc, dropna=False)[nc].mean().reset_index().head(30)
        figs.append((f"Mean {nc} by {cc}", px.bar(sub, x=cc, y=nc)))

    if num.shape[1] >= 2:
        figs.append(
            (
                f"Scatter: {num.columns[0]} vs {num.columns[1]}",
                px.scatter(df, x=num.columns[0], y=num.columns[1]),
            )
        )

    if len(cats) > 0:
        vc = df[cats[0]].value_counts().head(20).reset_index()
        vc.columns = [cats[0], "count"]
        figs.append((f"Top categories: {cats[0]}", px.bar(vc, x=cats[0], y="count")))

    return figs[:max_charts]


def answer_data_question_smart(
    client,
    df: pd.DataFrame,
    context: str,
    question: str,
    history: list[dict],
    lang_code: str,
    *,
    dict_index: Any = None,
    dict_chunks: Optional[list[dict[str, Any]]] = None,
    embed_model: Any = None,
) -> dict[str, Any]:
    """
    带路由与字典 RAG 的问答：train/text/schema 走专链，其余走结构化问答并启发式补图。
    """
    cols = df.columns.tolist()
    routed = route_data_intent(client, question, cols, lang_code)
    intent = routed.get("intent", "qa")

    if intent == "schema" and dict_index is not None and dict_chunks and embed_model is not None:
        text = answer_schema_rag(
            client, question, dict_index, dict_chunks, embed_model, history, lang_code
        )
        if not (text or "").strip():
            out = answer_data_question(
                client, df, context, question, history, lang_code
            )
            out["intent"] = intent
            out["route_note"] = routed.get("note")
            out["raw_json"] = {**(out.get("raw_json") or {}), "routed": routed, "intent": intent}
            return out
        return {
            "explanation": text,
            "figure": None,
            "table_df": None,
            "raw_json": {"intent": intent, "routed": routed},
            "intent": intent,
            "route_note": routed.get("note"),
        }

    if intent == "join":
        hint = (
            "请在下方「多表 Join」标签页上传至少两个表并选择关联键完成合并。"
            if lang_code == "zh"
            else "Use the **Multi-table join** tab to upload tables and set join keys."
        )
        return {
            "explanation": f"**[{intent}]** {routed.get('note', '')}\n\n{hint}",
            "figure": None,
            "table_df": None,
            "raw_json": {"intent": intent, "routed": routed},
            "intent": intent,
            "route_note": routed.get("note"),
        }

    if intent == "train" and _SKLEARN_OK:
        tgt = extract_column_name(
            client, question, cols, "prediction target column", lang_code
        )
        if tgt:
            tr = train_simple_model(df, tgt)
            if tr.get("ok"):
                expl = explain_training_result(client, tr, lang_code)
                fig_imp = importance_bar_figure(
                    tr.get("importance") or [], title=f"Importance → {tgt}"
                )
                metrics = tr.get("metrics") or {}
                md = f"**[train → {tgt}]** `{tr.get('task')}`\n\n- metrics: `{metrics}`\n\n{expl}"
                return {
                    "explanation": md,
                    "figure": fig_imp,
                    "table_df": None,
                    "raw_json": {"intent": intent, "train": tr, "routed": routed},
                    "intent": intent,
                    "route_note": routed.get("note"),
                }
            md_fail = f"**[train]** 无法训练：{tr.get('error', 'unknown')}。请在「建模」标签页手动选择目标列。"
            if lang_code != "zh":
                md_fail = f"**[train]** Cannot train: {tr.get('error', 'unknown')}. Use the **Modeling** tab."
            return {
                "explanation": md_fail,
                "figure": None,
                "table_df": None,
                "raw_json": {"intent": intent, "train": tr, "routed": routed},
                "intent": intent,
                "route_note": routed.get("note"),
            }

    if intent == "text":
        col = extract_column_name(
            client, question, cols, "text column to analyze", lang_code
        )
        if col and col in df.columns:
            prof = analyze_text_column(df[col], client, lang_code)
            terms = prof.get("top_terms") or []
            term_s = ", ".join(f"{a}({b:.0f})" for a, b in terms[:10])
            md = (
                f"**[text: {col}]** n={prof['n']}, 平均长度≈{prof['mean_len']:.1f}\n\n"
                f"{prof.get('summary', '')}\n\n"
                f"（词频 Top，英文 token）: {term_s or '—'}"
            )
            return {
                "explanation": md,
                "figure": None,
                "table_df": None,
                "raw_json": {"intent": intent, "profile": prof, "routed": routed},
                "intent": intent,
                "route_note": routed.get("note"),
            }

    out = answer_data_question(client, df, context, question, history, lang_code)
    fig = out.get("figure")
    if fig is None:
        h = heuristic_chart_spec(df, question)
        if h:
            out["figure"] = build_chart(df, h)
            out["chart_fallback"] = "heuristic"
    out["intent"] = intent
    out["route_note"] = routed.get("note")
    out["raw_json"] = {
        **(out.get("raw_json") or {}),
        "routed": routed,
        "intent": intent,
    }
    return out
