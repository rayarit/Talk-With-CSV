
## `streamlit_app.py`
import os
import re
import io
import sqlite3
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq

# ---------- App Config ----------
st.set_page_config(page_title="Discuss with your CSV (Groq)", page_icon="📊", layout="wide")
MODEL = "llama-3.1-8b-instant"  # Groq fast/free-tier friendly
FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|CREATE|TRIGGER)\b", re.I)

# ---------- Helpers ----------
# --- Make column names SQL-safe (snake_case, no spaces/symbols, no reserved words) ---
SQLITE_RESERVED = {
    "select","from","where","group","order","by","limit","offset","join","inner","left",
    "right","full","on","as","and","or","not","in","like","is","null","case","when","then",
    "else","end","union","all","distinct","having","between","exists","create","table",
    "index","insert","update","delete","drop","alter","values","into"
}

def clean_columns(df: pd.DataFrame):
    """
    Returns: (renamed_df, mapping: original -> cleaned)
    Rules:
      - lower_snake_case
      - remove non-alnum chars
      - prefix if starts with digit
      - avoid SQLite reserved words
      - dedupe by adding _1, _2...
    """
    mapping, used = {}, set()
    for c in df.columns:
        new = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_").lower()
        if new and new[0].isdigit():
            new = f"col_{new}"
        if new in SQLITE_RESERVED:
            new = f"{new}_col"
        base = new or "col"
        i = 1
        while new in used or new == "":
            new = f"{base}_{i}"
            i += 1
        used.add(new)
        mapping[c] = new
    return df.rename(columns=mapping), mapping


def get_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.info("Add GROQ_API_KEY in Streamlit Secrets to enable AI features.")
        return None
    return Groq(api_key=api_key)

def read_csv_safely(uploaded_file) -> pd.DataFrame:
    """Try utf-8 first, then latin-1; sample to 100k rows if huge."""
    bytes_data = uploaded_file.read()
    df = None
    for enc in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding=enc, low_memory=False)
            break
        except Exception:
            continue
    if df is None:
        raise ValueError("Could not read file as CSV. Please upload a valid CSV.")
    if len(df) > 100_000:
        df = df.sample(100_000, random_state=42).reset_index(drop=True)
        st.warning("Sampled 100,000 rows for performance.")
    return df

def cast_datetime_try(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort conversion for date-like text columns."""
    df2 = df.copy()
    for col in df2.columns:
        if df2[col].dtype == "object":
            sample = df2[col].dropna().astype(str).head(50).str.contains(r"[-/:\s]").mean()
            if sample > 0.3:
                try:
                    df2[col] = pd.to_datetime(df2[col], errors="ignore", infer_datetime_format=True)
                except Exception:
                    pass
    return df2

def basic_profile(df: pd.DataFrame) -> dict:
    info = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}
    summaries = []
    for c in df.columns:
        s = df[c]
        miss = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        col_info = {"column": c, "dtype": dtype, "missing_rate": round(miss, 4), "nunique": nunique}
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe(percentiles=[.05,.25,.5,.75,.95])
            col_info.update({
                "mean": float(desc.get("mean", np.nan)) if not pd.isna(desc.get("mean", np.nan)) else None,
                "std": float(desc.get("std", np.nan)) if not pd.isna(desc.get("std", np.nan)) else None,
                "min": float(desc.get("min", np.nan)) if not pd.isna(desc.get("min", np.nan)) else None,
                "p25": float(desc.get("25%", np.nan)) if not pd.isna(desc.get("25%", np.nan)) else None,
                "p50": float(desc.get("50%", np.nan)) if not pd.isna(desc.get("50%", np.nan)) else None,
                "p75": float(desc.get("75%", np.nan)) if not pd.isna(desc.get("75%", np.nan)) else None,
                "max": float(desc.get("max", np.nan)) if not pd.isna(desc.get("max", np.nan)) else None,
            })
        else:
            top_vals = s.dropna().astype(str).value_counts().head(5)
            col_info["top_values"] = top_vals.to_dict()
        summaries.append(col_info)
    info["columns"] = summaries
    return info

def profile_to_text(profile: dict, limit_cols: int = 30) -> str:
    lines = [f"Rows: {profile['n_rows']}, Cols: {profile['n_cols']}", f"Columns (up to {limit_cols}):"]
    for col in profile["columns"][:limit_cols]:
        line = f"- {col['column']} ({col['dtype']}), missing={col['missing_rate']}, nunique={col['nunique']}"
        if "mean" in col and col["mean"] is not None:
            line += f", mean={col['mean']}, p50={col['p50']}, max={col.get('max')}"
        elif "top_values" in col:
            line += f", top_values={list(col['top_values'].items())[:3]}"
        lines.append(line)
    return "\n".join(lines)

def get_schema_text(conn):
    q = "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name;"
    schema = pd.read_sql(q, conn)
    out = []
    for _, row in schema.iterrows():
        if row["sql"]:
            out.append(f"-- {row['name']}\n{row['sql']}")
    return "\n\n".join(out) if out else "-- [No tables]"

def extract_sql(text: str) -> str:
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = block.group(1).strip() if block else text.strip()
    if ";" in sql:
        sql = sql.split(";")[0] + ";"
    if not sql.endswith(";"):
        sql += ";"
    return sql

def is_safe_sql(sql: str) -> bool:
    return sql.strip().lower().startswith("select") and not FORBIDDEN.search(sql)

# ---------- LLM Calls ----------
def llm_ai_insights(client: Groq, profile_text: str, sample_md: str) -> str:
    system = (
        "You are a sharp data analyst. Given a dataset profile and a small sample, "
        "write 5-8 bullet insights: distributions, missingness, potential data issues, "
        "interesting groupings, and 2 follow-up questions. Be concrete."
    )
    user = f"Dataset profile:\n{profile_text}\n\nSample rows (markdown table):\n{sample_md}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.3
    )
    return resp.choices[0].message.content

def llm_sql(client: Groq, schema_text: str, question: str) -> str:
    system = (
        "You are a careful data analyst. Given an SQLite schema and a user question, "
        "produce a SINGLE, safe, SELECT-only SQLite query that answers it. "
        "Use the exact table/column names from the schema (they are snake_case). "
        "No PRAGMA, no temp tables, no DDL/DML. Return ONLY SQL in a code block."
    )
    user = f"SQLite schema:\n{schema_text}\n\nUser question: {question}\n\nReturn ONLY SQL."
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0
    )
    return resp.choices[0].message.content


def llm_explain(client: Groq, question: str, sql: str, sample_df: pd.DataFrame) -> str:
    sys = "Explain the result succinctly for a non-technical stakeholder in 3–6 bullets."
    sample_md = sample_df.head(5).to_markdown(index=False)
    user = f"Question: {question}\n\nSQL:\n{sql}\n\nFirst 5 result rows:\n{sample_md}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ---------- UI ----------
st.title("📊 Discuss with your CSV with Groq")
st.caption("Upload a CSV → auto-insights → ask questions in plain English (converted to safe SQL).")

if "df" not in st.session_state:
    st.session_state.df = None

left, right = st.columns([1, 1])
with left:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    use_example = st.checkbox("Or load a tiny example dataset", value=False)
with right:
    st.markdown("**Tips**")
    st.markdown(
        "- For large files, the app will sample 100k rows.\n"
        "- Try questions like: *“total sales by region this year”*, *“top 10 categories by avg price”*."
    )

# Load data
if uploaded or use_example:
    try:
        if uploaded:
            df_raw = read_csv_safely(uploaded)
        else:
            df_raw = pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=60, freq="D"),
                "region": np.random.choice(["East","West","North","South"], size=60),
                "category": np.random.choice(["A","B","C"], size=60),
                "amount": np.random.gamma(3., 50., size=60).round(2),
                "units": np.random.poisson(5, size=60)
            })
        df = cast_datetime_try(df_raw)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Failed to load file: {e}")

# Preview & stats
if st.session_state.df is not None:
    df = st.session_state.df
    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Quick stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Cols with any missing", f"{(df.isna().any().mean() * 100):.1f}%")

    with st.expander("Column summary"):
        col_summary = pd.DataFrame({
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_%": (df.isna().mean() * 100).round(2).values,
            "nunique": [df[c].nunique(dropna=True) for c in df.columns]
        })
        st.dataframe(col_summary, use_container_width=True)

    with st.expander("Quick visual"):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            sel_num = st.selectbox("Histogram column", numeric_cols)
            vals = df[sel_num].dropna().values
            hist, bin_edges = np.histogram(vals, bins=20)
            hist_df = pd.DataFrame({"bin_left": bin_edges[:-1], "count": hist})
            st.bar_chart(hist_df, x="bin_left", y="count", use_container_width=True)
        cat_cols = [c for c in df.columns if df[c].dtype == "object" or pd.api.types.is_categorical_dtype(df[c])]
        if cat_cols:
            sel_cat = st.selectbox("Top categories", cat_cols)
            vc = df[sel_cat].astype(str).value_counts().head(15)
            st.bar_chart(vc)

    # AI Insights
    st.subheader("AI Insights")
    client = get_client()
    if client:
        if st.button("Generate insights from this dataset"):
            with st.spinner("Thinking..."):
                profile = basic_profile(df)
                profile_text = profile_to_text(profile)
                sample_md = df.head(10).to_markdown(index=False)
                insights = llm_ai_insights(client, profile_text, sample_md)
            st.markdown(insights)
    else:
        st.info("Add GROQ_API_KEY in Secrets to enable AI insights.")

    # Q&A
    st.subheader("Ask questions about your data")
    q = st.text_input("Your question (e.g., Total amount by region for 2025?)")
    run = st.button("Generate SQL and run")

    if run:
        if not q.strip():
            st.warning("Please type a question.")
            st.stop()

        # conn = sqlite3.connect(":memory:")
        # table_name = "data"
        # df.to_sql(table_name, conn, if_exists="replace", index=False)
        # schema_text = get_schema_text(conn)
        # st.code(schema_text, language="sql")
        conn = sqlite3.connect(":memory:")
        table_name = "data"

        # ✨ sanitize columns before loading to SQLite
        df_sql, colmap = clean_columns(df)
        df_sql.to_sql(table_name, conn, if_exists="replace", index=False)

        # Show mapping so users know what to ask for
        with st.expander("Column name mapping (original → SQL-safe)"):
            map_df = pd.DataFrame({"original": list(colmap.keys()), "sql_name": list(colmap.values())})
            st.dataframe(map_df.sort_values("original"), use_container_width=True)

        schema_text = get_schema_text(conn)
        st.code(schema_text, language="sql")


        if not client:
            st.info("Add GROQ_API_KEY in Secrets to enable Q&A.")
            st.stop()

        with st.spinner("Composing a safe SQL query..."):
            draft_sql = llm_sql(client, schema_text, q)
            sql = extract_sql(draft_sql)

        st.markdown("**Generated SQL**")
        st.code(sql, language="sql")

        if not is_safe_sql(sql):
            st.error("Generated SQL not considered safe (SELECT-only). Try rephrasing your question.")
            st.stop()

        try:
            result_df = pd.read_sql_query(sql, conn)
        except Exception as e:
            st.error(f"SQL error: {e}")
            st.stop()

        st.subheader("Results")
        st.dataframe(result_df, use_container_width=True)

        # Download
        if len(result_df) > 0:
            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", data=csv_bytes, file_name="query_results.csv", mime="text/csv")

        # Quick chart
        num_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        if len(num_cols) >= 1 and len(result_df) > 0:
            with st.expander("Quick chart"):
                x_col = st.selectbox("X", result_df.columns, index=0)
                y_col = st.selectbox("Y (numeric)", num_cols, index=0)
                try:
                    st.bar_chart(result_df.set_index(x_col)[y_col])
                except Exception:
                    st.line_chart(result_df[y_col])

        # Explanation
        if len(result_df) > 0 and client:
            with st.spinner("Explaining the results..."):
                explanation = llm_explain(client, q, sql, result_df)
            st.subheader("Explanation")
            st.markdown(explanation)
