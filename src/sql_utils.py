import re
import pandas as pd

FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|REPLACE|CREATE|TRIGGER)\b", re.I)

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
