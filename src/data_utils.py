import io
import numpy as np
import pandas as pd
import streamlit as st

def read_csv_safely(uploaded_file) -> pd.DataFrame:
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
