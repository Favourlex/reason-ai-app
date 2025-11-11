# ================================================
# 🟧 Reason AI Backend API for NeuralSeek
# FastAPI endpoint that answers REST queries
# Reads artifacts/cleaned_latest.csv from Streamlit
# ================================================
from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Reason AI API", version="1.0.0")

# ---------- Helper ----------
def detect_numeric_series(s: pd.Series) -> bool:
    from pandas.api.types import is_numeric_dtype
    if is_numeric_dtype(s): return True
    coerce = pd.to_numeric(s.dropna().astype(str).str.replace(",", ""), errors="coerce")
    return (coerce.notna().mean() >= 0.7)

def to_numeric_smart(s: pd.Series) -> pd.Series:
    cleaned = (
        s.astype(str)
         .str.replace(r"[₦$,()% ]", "", regex=True)
         .str.replace(r"[^\d\.\-eE]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")

def find_first(df, *keys):
    cols = list(df.columns)
    keys_low = [k.lower() for k in keys]
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in keys_low):
            return c
    return None

# ---------- Core endpoint ----------
@app.get("/query", response_class=PlainTextResponse)
def query(q: str = Query(..., description="Question title from NeuralSeek")):
    data_path = "artifacts/cleaned_latest.csv"
    if not os.path.exists(data_path):
        return "⚠️ No cleaned dataset available. Please run Streamlit app first."

    df = pd.read_csv(data_path)
    if df.empty:
        return "⚠️ Dataset is empty."

    doctor_col  = find_first(df, "doctor", "physician", "consultant")
    clinic_col  = find_first(df, "clinic", "location", "ward", "department")
    fee_col     = find_first(df, "fee", "amount", "charge", "bill", "cost", "price")
    date_col    = find_first(df, "date", "admission", "discharge", "visit", "created")
    gender_col  = find_first(df, "gender", "sex")
    age_col     = find_first(df, "age")
    outcome_col = find_first(df, "outcome", "status", "result")

    if fee_col is not None:
        try: df[fee_col] = to_numeric_smart(df[fee_col])
        except Exception: pass

    q_lower = q.lower()
    if "doctor" in q_lower:
        if doctor_col:
            vc = df[doctor_col].replace("Unknown", np.nan).dropna().value_counts()
            if vc.empty: return "No doctor data."
            return f"Total doctors: {vc.index.nunique()} | Top doctor: {vc.index[0]} ({int(vc.iloc[0])} records)"
        return "Doctor column missing."

    if "clinic" in q_lower:
        if clinic_col:
            vc = df[clinic_col].replace("Unknown", np.nan).dropna().value_counts()
            if vc.empty: return "No clinic data."
            return f"Total clinics: {vc.index.nunique()} | Busiest clinic: {vc.index[0]} ({int(vc.iloc[0])} records)"
        return "Clinic column missing."

    if "revenue" in q_lower or "fee" in q_lower:
        if date_col and fee_col:
            d = pd.to_datetime(df[date_col].replace("Unknown", np.nan), errors="coerce")
            tmp = pd.DataFrame({"month": d.dt.to_period("M").astype(str), "fee": df[fee_col]})
            tmp = tmp.dropna()
            gp = tmp.groupby("month")["fee"].sum().sort_index()
            if gp.empty: return "No date/fee data."
            return f"Revenue window: {gp.index.min()} → {gp.index.max()} | Peak month: {gp.idxmax()} (₦{gp.max():,.0f})"
        return "Fee or date column missing."

    if "gender" in q_lower:
        if gender_col:
            vc = df[gender_col].replace("Unknown", np.nan).dropna().value_counts()
            if vc.empty: return "No gender data."
            return "Gender mix: " + ", ".join([f"{k}: {int(v)}" for k, v in vc.items()])
        return "Gender column missing."

    if "age" in q_lower:
        if age_col and detect_numeric_series(df[age_col]):
            s = to_numeric_smart(df[age_col]).dropna()
            if s.empty: return "No age data."
            return f"Age — mean: {s.mean():.1f}, median: {s.median():.1f}, n={len(s)}"
        return "Age column missing."

    return "Unsupported question or columns missing."
