import os
import pandas as pd
from groq import Groq
import streamlit as st

MODEL = "llama-3.1-8b-instant"

def get_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.info("Add GROQ_API_KEY in Streamlit Secrets to enable AI features.")
        return None
    return Groq(api_key=api_key)

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
    user = f"Question: {question}\n\nSQL:\n{sql}\n\nFirst 5 result rows:\n{sample_df.head(5).to_markdown(index=False)}"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        temperature=0.2
    )
    return resp.choices[0].message.content
