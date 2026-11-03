"""
Tabular data analysis: load CSV/Excel, rule-based cleaning, LLM overview/quality/insights,
structured JSON Q&A with safe Plotly charts and pandas table operations, and Markdown report export.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Optional

import pandas as pd
import plotly.express as px

MODEL = "deepseek-chat"

LANG_NAMES = {
    "zh": "中文",
    "en": "English",
    "de": "Deutsch",
    "fr": "Français",
}


from src.api import request


def _chat(client, messages: list[dict], temperature: float = 0.2) -> str:
    r = request(client,
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return (r.choices[0].message.content or "").strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(text[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def auto_clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    规则化自动清洗。返回 (清洗后 DataFrame, 步骤列表)。
    每步为 {"key": "...", ...}，供 UI 用 t('da_log_'+key) 展示。
    """
    out = df.copy()
    steps: list[dict[str, Any]] = []

    # 1) 字符串列去首尾空格；常见空占位
    obj_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    if len(obj_cols) > 0:
        for c in obj_cols:
            out[c] = out[c].astype(str).str.strip()
            out[c] = out[c].replace(
                {"", "nan", "None", "NaN", "NULL", "null", "<NA>"},
                pd.NA,
            )
        steps.append({"key": "whitespace_stripped", "n_cols": len(obj_cols)})

    # 2) 删除完全重复行
    before = len(out)
    out = out.drop_duplicates()
    dup_n = before - len(out)
    if dup_n > 0:
        steps.append({"key": "duplicates_removed", "n": dup_n})

    # 3) 删除全空列
    empty_cols = [c for c in out.columns if out[c].isna().all()]
    if empty_cols:
        out = out.drop(columns=empty_cols)
        steps.append({"key": "empty_cols_dropped", "n": len(empty_cols)})

    # 4) 尝试将可转换 object 列为数值
    converted = []
    for c in out.select_dtypes(include=["object"]).columns:
        try:
            conv = pd.to_numeric(out[c], errors="coerce")
            if conv.notna().sum() >= max(3, int(0.6 * len(out))):
                out[c] = conv
                converted.append(c)
        except Exception:
            pass
    if converted:
        cols_s = ", ".join(converted[:8]) + ("…" if len(converted) > 8 else "")
        steps.append({"key": "converted_numeric", "cols": cols_s})

    # 5) 数值列：缺失用中位数
    num_cols = out.select_dtypes(include="number").columns.tolist()
    filled_n = 0
    for c in num_cols:
        if out[c].isna().any():
            med = out[c].median()
            if pd.isna(med):
                out[c] = out[c].fillna(0)
            else:
                out[c] = out[c].fillna(med)
            filled_n += 1
    if filled_n > 0:
        steps.append({"key": "numeric_median_fill", "n_cols": filled_n})

    # 6) 非数值列：缺失用众数或占位
    num_cols_final = out.select_dtypes(include="number").columns.tolist()
    other = [c for c in out.columns if c not in num_cols_final]
    cat_filled = 0
    for c in other:
        if out[c].isna().any():
            mode = out[c].mode(dropna=True)
            fill_v = mode.iloc[0] if len(mode) else "Unknown"
            out[c] = out[c].fillna(fill_v)
            cat_filled += 1
    if cat_filled > 0:
        steps.append({"key": "categorical_fill", "n_cols": cat_filled})

    return out, steps


def build_statistical_profile(df: pd.DataFrame) -> dict[str, Any]:
    """结构化统计分析结果（用于展示与喂给 LLM）。"""
    num = df.select_dtypes(include="number")
    profile: dict[str, Any] = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "nunique": df.nunique().to_dict(),
        "describe_all": df.describe(include="all").to_string(),
    }
    if len(num.columns) > 0:
        profile["skew"] = num.skew(numeric_only=True).to_string()
        if len(num.columns) >= 2:
            profile["corr"] = num.corr().to_string()
        else:
            profile["corr"] = ""
    else:
        profile["skew"] = ""
        profile["corr"] = ""
    text_blob = (
        f"shape: {profile['shape']}\n\n"
        f"null_counts:\n{df.isnull().sum().to_string()}\n\n"
        f"nunique:\n{pd.Series(profile['nunique']).to_string()}\n\n"
        f"describe:\n{profile['describe_all']}\n\n"
    )
    if profile["skew"]:
        text_blob += f"skewness:\n{profile['skew']}\n\n"
    if profile["corr"]:
        text_blob += f"correlation (numeric):\n{profile['corr']}\n"
    profile["text_for_llm"] = text_blob[:20000]
    return profile


def correlation_heatmap_figure(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    c = num.corr()
    return px.imshow(
        c,
        text_auto=".2f",
        aspect="auto",
        title="Correlation",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )


def generate_cleaning_explanation(
    client,
    steps: list[dict[str, Any]],
    before_rows: int,
    after_rows: int,
    lang_code: str,
) -> str:
    lang = LANG_NAMES.get(lang_code, "中文")
    lines = [str(s) for s in steps]
    system = (
        f"用{lang}写一段话（3～6 句），说明对数据做了哪些清洗、清洗前后行数变化，"
        "面向业务读者，不要出现 JSON。"
    )
    user = (
        f"清洗前行数: {before_rows}, 清洗后行数: {after_rows}\n"
        f"步骤（结构化）:\n" + "\n".join(lines)
    )
    return _chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.25)


def generate_insight_summary(
    client,
    context: str,
    stats_text: str,
    lang_code: str,
) -> str:
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是高级数据分析师。根据数据摘要与统计信息，用{lang}写「洞察总结」："
        "3～8 条要点，关注趋势、异常、字段关系、业务含义与可行动建议；使用 Markdown 列表。"
        "不要编造上下文中没有依据的结论。"
    )
    user = f"--- 摘要 ---\n{context[:10000]}\n\n--- 统计 ---\n{stats_text[:10000]}"
    return _chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.35)


def load_uploaded_table(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
    bio = io.BytesIO(raw)
    if name.endswith(".csv"):
        return pd.read_csv(bio)
    if name.endswith(".xlsx"):
        return pd.read_excel(bio, engine="openpyxl")
    if name.endswith(".xls"):
        return pd.read_excel(bio)
    raise ValueError("Unsupported file type (use .csv, .xlsx, .xls)")


def dataframe_context_for_llm(df: pd.DataFrame, max_sample_rows: int = 25) -> str:
    buf: list[str] = []
    buf.append(f"shape: {df.shape[0]} rows × {df.shape[1]} columns")
    buf.append("\n dtypes:\n" + df.dtypes.to_string())
    buf.append("\n null counts:\n" + df.isnull().sum().to_string())
    numeric = df.select_dtypes(include='number')
    if not numeric.empty:
        buf.append("\n numeric describe:\n" + numeric.describe().to_string())
    try:
        buf.append("\n categorical / object top values (sample):")
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype).startswith("category"):
                vc = df[col].value_counts(dropna=False).head(8)
                buf.append(f"\n[{col}]\n{vc.to_string()}")
    except Exception:
        pass
    buf.append("\n\n sample rows (first rows):\n")
    buf.append(df.head(max_sample_rows).to_csv(index=False))
    return "\n".join(buf)[:24000]


def generate_overview(client, context: str, lang_code: str) -> str:
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是数据分析师。根据下列表格统计摘要，用{lang}写一段「数据概览」："
        "行数列数、主要字段含义（推断）、数值/类别分布要点。不要编造具体数字以外的内容。"
    )
    user = f"--- 数据摘要 ---\n{context}\n---\n请输出简洁的概览（可用 Markdown 小标题）。"
    return _chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.3)


def generate_quality_issues(client, context: str, lang_code: str) -> str:
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是数据质量顾问。根据摘要用{lang}列出可能的数据质量问题："
        "缺失值、重复、异常值风险、类型不一致、类别不平衡等。条目化，避免空话。"
    )
    user = f"--- 数据摘要 ---\n{context}\n---\n输出问题清单（Markdown 列表）。"
    return _chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.2)


def _validate_cols(df: pd.DataFrame, cols: list[str] | None) -> bool:
    if not cols:
        return True
    return all(c in df.columns for c in cols)


def run_table_op(df: pd.DataFrame, spec: dict[str, Any]) -> Optional[pd.DataFrame]:
    if not spec or not isinstance(spec, dict):
        return None
    op = spec.get("op")
    try:
        if op == "head":
            n = int(spec.get("n", 20))
            return df.head(max(1, min(n, 500)))
        if op == "tail":
            n = int(spec.get("n", 20))
            return df.tail(max(1, min(n, 500)))
        if op == "describe":
            return df.describe(include="all").transpose()
        if op == "value_counts":
            col = spec.get("by") or spec.get("column")
            if col not in df.columns:
                return None
            top = int(spec.get("top", 20))
            return df[col].value_counts(dropna=False).head(top).rename("count").reset_index()
        if op == "groupby_agg":
            gcols = spec.get("group_cols") or spec.get("by")
            agg = spec.get("agg") or {}
            if not gcols or not isinstance(gcols, list) or not _validate_cols(df, gcols):
                return None
            metrics = {}
            for k, v in agg.items():
                if k not in df.columns:
                    continue
                if isinstance(v, list):
                    metrics[k] = v[0] if v else "sum"
                else:
                    metrics[k] = v
            if not metrics:
                return None
            return df.groupby(gcols, dropna=False).agg(metrics).reset_index()
        if op == "corr":
            return df.select_dtypes(include="number").corr()
        if op == "sort_values":
            by = spec.get("by")
            if by not in df.columns:
                return None
            asc = bool(spec.get("ascending", False))
            n = int(spec.get("n", 50))
            return df.sort_values(by=by, ascending=asc).head(max(1, min(n, 500)))
    except Exception:
        return None
    return None


def build_chart(df: pd.DataFrame, spec: dict[str, Any]):
    if not spec or not isinstance(spec, dict):
        return None
    ctype = (spec.get("type") or "").lower()
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")

    def _safe_col(c):
        return c if c in df.columns else None

    try:
        if ctype == "histogram":
            xc = _safe_col(x)
            if not xc:
                return None
            return px.histogram(df, x=xc, color=_safe_col(color) if color else None)

        if ctype == "bar":
            g = spec.get("group_cols")
            agg = spec.get("agg")
            if g and isinstance(g, list) and agg and _validate_cols(df, g):
                metrics = {k: v for k, v in agg.items() if k in df.columns}
                if metrics:
                    plot_df = df.groupby(g, dropna=False).agg(metrics).reset_index()
                    yc = list(metrics.keys())[0]
                    return px.bar(plot_df, x=g[0] if len(g) == 1 else g[0], y=yc)
            xc, yc = _safe_col(x), _safe_col(y)
            if xc and yc:
                return px.bar(df, x=xc, y=yc, color=_safe_col(color) if color else None)
            if xc:
                vc = df[xc].value_counts().head(30).reset_index()
                vc.columns = [xc, "count"]
                return px.bar(vc, x=xc, y="count")

        if ctype == "line":
            xc, yc = _safe_col(x), _safe_col(y)
            if xc and yc:
                return px.line(df.sort_values(xc), x=xc, y=yc, color=_safe_col(color) if color else None)

        if ctype == "scatter":
            xc, yc = _safe_col(x), _safe_col(y)
            if xc and yc:
                return px.scatter(df, x=xc, y=yc, color=_safe_col(color) if color else None)

        if ctype == "box":
            xc = _safe_col(x)
            yc = _safe_col(y)
            if xc and yc:
                return px.box(df, x=xc, y=yc)
            if yc:
                return px.box(df, y=yc)
    except Exception:
        return None
    return None


def answer_data_question(
    client,
    df: pd.DataFrame,
    context: str,
    question: str,
    history: list[dict],
    lang_code: str,
) -> dict[str, Any]:
    lang = LANG_NAMES.get(lang_code, "中文")
    hist_lines = []
    for m in history[-8:]:
        role = m.get("role", "")
        hist_lines.append(f"{role}: {m.get('content', '')[:1500]}")
    hist_text = "\n".join(hist_lines) if hist_lines else "(none)"

    schema = {c: str(dt) for c, dt in df.dtypes.items()}

    system = f"""你是数据分析助手。用户用{lang}提问。你必须只输出一个 JSON 对象（不要 markdown 围栏外其它文字），结构如下：
{{
  "explanation": "用{lang}写的分析说明，支持 Markdown。",
  "chart": null 或 {{
    "type": "histogram|bar|line|scatter|box",
    "x": "列名",
    "y": "列名或省略",
    "color": "列名或省略",
    "group_cols": ["分组列"],
    "agg": {{"数值列": "sum|mean|count|min|max"}}
  }},
  "table": null 或 {{
    "op": "head|tail|describe|value_counts|groupby_agg|corr|sort_values",
    "n": 20,
    "by": "列名",
    "top": 20,
    "group_cols": [],
    "agg": {{}},
    "ascending": false
  }}
}}
规则：列名必须来自提供的 schema，禁止编造列名。若不适合作图，chart 用 null。"""

    user = f"""schema (列→类型):\n{json.dumps(schema, ensure_ascii=False)}\n\n数据摘要:\n{context[:12000]}\n\n最近对话:\n{hist_text}\n\n用户问题:\n{question}"""

    raw = _chat(
        client,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        0.25,
    )
    data = _parse_json_object(raw) or {}
    explanation = data.get("explanation") or raw
    chart_spec = data.get("chart")
    table_spec = data.get("table")

    table_df = None
    if isinstance(table_spec, dict):
        table_df = run_table_op(df, table_spec)

    fig = None
    if isinstance(chart_spec, dict) and chart_spec:
        fig = build_chart(df, chart_spec)

    return {
        "explanation": explanation,
        "figure": fig,
        "table_df": table_df,
        "raw_json": data,
    }


def generate_analysis_report(
    client,
    overview: str,
    quality: str,
    context: str,
    chat_snippets: str,
    lang_code: str,
    *,
    cleaning_explanation: str = "",
    insights: str = "",
    stats_excerpt: str = "",
) -> str:
    lang = LANG_NAMES.get(lang_code, "中文")
    system = (
        f"你是专业报告撰写人。用{lang}写一份「数据分析报告」，结构建议："
        "一、数据说明与清洗摘要 二、统计与分布概览 三、数据质量与风险 四、洞察总结要点 五、对话中的主要发现 六、建议与后续工作。"
        "语气正式、条理清晰，使用 Markdown。"
    )
    user = (
        f"【清洗说明】\n{cleaning_explanation[:4000]}\n\n"
        f"【自动概览】\n{overview[:4000]}\n\n【质量问题】\n{quality[:4000]}\n\n"
        f"【洞察总结】\n{insights[:4000]}\n\n"
        f"【统计摘要】\n{stats_excerpt[:6000]}\n\n"
        f"【数据上下文片段】\n{context[:6000]}\n\n【用户问答摘录】\n{chat_snippets[:5000]}"
    )
    return _chat(client, [{"role": "system", "content": system}, {"role": "user", "content": user}], 0.35)


def build_export_python_script(filename: str) -> str:
    """与 auto_clean_dataframe 逻辑对齐的可运行模板（需自行改读文件路径）。"""
    return f'''# -*- coding: utf-8 -*-
"""Data pipeline template — mirrors in-app auto-cleaning.
Replace FILE with your CSV/XLSX path.
"""
import pandas as pd
import plotly.express as px

FILE = "{filename}"  # change to your path

def auto_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    oc = out.select_dtypes(include=["object", "string"]).columns
    for c in oc:
        out[c] = out[c].astype(str).str.strip()
        out[c] = out[c].replace({{"", "nan", "None", "NaN", "NULL", "null", "<NA>"}}, pd.NA)
    out = out.drop_duplicates()
    ec = [c for c in out.columns if out[c].isna().all()]
    if ec:
        out = out.drop(columns=ec)
    for c in out.select_dtypes(include=["object"]).columns:
        conv = pd.to_numeric(out[c], errors="coerce")
        if conv.notna().sum() >= max(3, int(0.6 * len(out))):
            out[c] = conv
    num_cols = out.select_dtypes(include="number").columns
    for c in num_cols:
        if out[c].isna().any():
            med = out[c].median()
            out[c] = out[c].fillna(0 if pd.isna(med) else med)
    num_f = out.select_dtypes(include="number").columns.tolist()
    for c in [x for x in out.columns if x not in num_f]:
        if out[c].isna().any():
            mo = out[c].mode(dropna=True)
            out[c] = out[c].fillna(mo.iloc[0] if len(mo) else "Unknown")
    return out

def main():
    # df = pd.read_csv(FILE)
    # df = pd.read_excel(FILE)
    # df2 = auto_clean(df)
    # print(df2.describe(include="all"))
    # num = df2.select_dtypes(include="number")
    # if num.shape[1] >= 2:
    #     fig = px.imshow(num.corr(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
    #     fig.write_html("correlation.html")
    pass

if __name__ == "__main__":
    main()
'''
