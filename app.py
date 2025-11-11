# ================================================
# 🧠 Reason AI — Universal Data Cleaner
# + Fixed-Question Local Q&A (Ask vs Visualize)
# + NeuralSeek Console Upload (unchanged)
# + Single-file API Mode (?api=1) for NeuralSeek REST
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, re, time
from io import BytesIO
from datetime import datetime
import requests
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reason AI - Universal Data Cleaner", layout="wide")

# ---------- Header ----------
st.markdown("""
<div style="text-align:left; margin-bottom:20px;">
  <h1 style="font-size:40px; font-weight:700; margin:0;">🧠 Reason AI</h1>
  <div style="font-size:18px; color:#444; font-weight:600; margin:2px 0 8px 0;">
    Developed by <span style="color:#1f77b4;"><b>Favour Ezeofor</b></span>
  </div>
  <h2 style="font-size:26px; font-weight:700; margin:0;">Universal Data Cleaning System</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload any **CSV file** — Reason AI will automatically clean it:

- Remove empty rows/columns  
- Handle missing values (smart numeric + text)  
- Detect and normalize dates  
- Standardize gender markers  
- Output a clean CSV ready for NeuralSeek ingestion  
- Generate a full data-quality summary  
""")

# ---------- Cleaning helpers ----------
NULL_TOKENS = {"", " ", "nan", "none", "null", "n/a", "na", "unknown"}

GENDER_MAP = {
    "m":"Male","male":"Male","man":"Male","boy":"Male",
    "f":"Female","female":"Female","woman":"Female","girl":"Female",
    "others":"Other","other":"Other","non-binary":"Other","nonbinary":"Other","nb":"Other",
    "trans":"Other","transgender":"Other"
}

def clean_text_cell(x: str) -> str:
    if pd.isna(x): return "Unknown"
    s = str(x).strip()
    return "Unknown" if s.lower() in NULL_TOKENS else s

def detect_numeric_series(s: pd.Series) -> bool:
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

def normalize_date_series(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).map(lambda x: x if x.strip().lower() not in NULL_TOKENS else np.nan)
    parsed = pd.to_datetime(cleaned, errors="coerce", infer_datetime_format=True)
    out = parsed.dt.strftime("%Y-%m-%d")
    return out.fillna("Unknown")

def clean_gender_cell(x: str) -> str:
    s = str(x).strip().lower()
    if s in NULL_TOKENS: return "Unknown"
    return GENDER_MAP.get(s, "Unknown")

@st.cache_data(show_spinner=False)
def _read_csv_cached(file_bytes: bytes, name: str):
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=enc), enc
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(file_bytes)), "auto"

# ---------- SINGLE-FILE API MODE (for NeuralSeek REST) ----------
def _api_mode_if_requested():
    try:
        qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
    except Exception:
        qp = {}

    api_flag = None
    if isinstance(qp, dict):
        api_flag = qp.get("api")
        if isinstance(api_flag, list): api_flag = api_flag[0]
    if str(api_flag).lower() not in ("1", "true", "yes", "query"):
        return  # UI mode continues

    def _find_first(cols, *keys):
        keys_low = [k.lower() for k in keys]
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in keys_low): return c
        return None

    def _month_series(df: pd.DataFrame, c: str):
        try:
            d = pd.to_datetime(df[c].replace("Unknown", np.nan), errors="coerce")
            return d.dt.to_period("M").dropna().astype(str)
        except Exception:
            return pd.Series(dtype="object")

    # Load DF
    df = st.session_state.get("df_clean")
    if df is None:
        try:
            if os.path.isdir("artifacts"):
                files = [f for f in os.listdir("artifacts") if f.endswith(".csv")]
                files.sort(key=lambda f: os.path.getmtime(os.path.join("artifacts", f)), reverse=True)
                if files:
                    df = pd.read_csv(os.path.join("artifacts", files[0]))
        except Exception:
            df = None

    def _first(name, default=""):
        v = qp.get(name, [default])
        return v[0] if isinstance(v, list) else v

    export = _first("export").lower().strip()
    fmt    = _first("format", "json").lower().strip()
    q      = _first("q") or _first("question")

    if export == "csv":
        if df is None or df.empty:
            st.text("No cleaned dataset available. Clean a CSV first."); st.stop()
        limit = _first("limit") or _first("n")
        try: limit = int(limit) if str(limit).strip() else 0
        except Exception: limit = 0
        out_df = df.head(limit) if limit and limit > 0 else df
        st.text(out_df.to_csv(index=False)); st.stop()

    if df is None or df.empty:
        st.text("No cleaned dataset available. Clean a CSV first."); st.stop()

    cols = list(df.columns)
    doctor_col  = _find_first(cols, "doctor","physician","consultant")
    clinic_col  = _find_first(cols, "clinic","location","ward","department")
    fee_col     = _find_first(cols, "fee","amount","charge","bill","cost","price")
    date_col    = _find_first(cols, "date","admission","discharge","visit","created")
    gender_col  = _find_first(cols, "gender","sex")
    age_col     = _find_first(cols, "age")
    proc_col    = _find_first(cols, "procedure","treatment","service")
    outcome_col = _find_first(cols, "outcome","readmit","readmission","status","result")

    if fee_col is not None:
        try: df[fee_col] = to_numeric_smart(df[fee_col])
        except Exception: pass

    CATALOG = []
    if doctor_col: CATALOG.append("Top doctors by patient count")
    if clinic_col: CATALOG.append("Top clinics by patient count")
    if fee_col and doctor_col: CATALOG.append("Average fee by doctor (Top 8)")
    if fee_col and clinic_col: CATALOG.append("Average fee by clinic (Top 8)")
    if date_col: CATALOG.append("Monthly case trend")
    if date_col and fee_col: CATALOG.append("Monthly revenue (sum of fee)")
    if gender_col: CATALOG.append("Gender distribution")
    if age_col and detect_numeric_series(df[age_col]): CATALOG.append("Age distribution (histogram)")
    if proc_col: CATALOG.append("Top procedures by count")
    if outcome_col and clinic_col: CATALOG.append("Readmission/Adverse rate by clinic")

    def _compute_answer(title: str):
        if title == "Top doctors by patient count" and doctor_col:
            vc = df[doctor_col].replace("Unknown", np.nan).dropna().value_counts()
            return "No doctor data." if vc.empty else f"Total doctors: {vc.index.nunique()} | Top doctor: {vc.index[0]} ({int(vc.iloc[0])} records)"
        if title == "Top clinics by patient count" and clinic_col:
            vc = df[clinic_col].replace("Unknown", np.nan).dropna().value_counts()
            return "No clinic data." if vc.empty else f"Total clinics: {vc.index.nunique()} | Busiest clinic: {vc.index[0]} ({int(vc.iloc[0])} records)"
        if title == "Average fee by doctor (Top 8)" and fee_col and doctor_col:
            tmp = df[[doctor_col, fee_col]].dropna()
            gp = tmp.groupby(doctor_col)[fee_col].mean().sort_values(ascending=False)
            return "No fee/doctor data." if gp.empty else f"Highest avg fee: {gp.index[0]} (₦{gp.iloc[0]:,.0f})"
        if title == "Average fee by clinic (Top 8)" and fee_col and clinic_col:
            tmp = df[[clinic_col, fee_col]].dropna()
            gp = tmp.groupby(clinic_col)[fee_col].mean().sort_values(ascending=False)
            return "No fee/clinic data." if gp.empty else f"Highest avg fee clinic: {gp.index[0]} (₦{gp.iloc[0]:,.0f})"
        if title == "Monthly case trend" and date_col:
            m = _month_series(df, date_col)
            s = m.value_counts().sort_index()
            return "No date data." if s.empty else f"Monthly range: {s.index.min()} → {s.index.max()} | Peak: {s.idxmax()} ({int(s.max())})"
        if title == "Monthly revenue (sum of fee)" and date_col and fee_col:
            d = pd.to_datetime(df[date_col].replace("Unknown", np.nan), errors="coerce")
            tmp = pd.DataFrame({"month": d.dt.to_period("M").astype(str), "fee": df[fee_col]})
            tmp = tmp.dropna()
            gp = tmp.groupby("month")["fee"].sum().sort_index()
            return "No date/fee data." if gp.empty else f"Revenue window: {gp.index.min()} → {gp.index.max()} | Peak month: {gp.idxmax()} (₦{gp.max():,.0f})"
        if title == "Gender distribution" and gender_col:
            vc = df[gender_col].replace("Unknown", np.nan).dropna().value_counts()
            return "No gender data." if vc.empty else "Gender mix: " + ", ".join([f"{k}: {int(v)}" for k,v in vc.items()])
        if title == "Age distribution (histogram)" and age_col and detect_numeric_series(df[age_col]):
            s = to_numeric_smart(df[age_col]).dropna()
            return "No age data." if s.empty else f"Age — mean: {s.mean():.1f}, median: {s.median():.1f}, n={len(s)}"
        if title == "Top procedures by count" and proc_col:
            vc = df[proc_col].replace("Unknown", np.nan).dropna().value_counts()
            return "No procedure data." if vc.empty else f"Top procedure: {vc.index[0]} ({int(vc.iloc[0])})"
        if title == "Readmission/Adverse rate by clinic" and outcome_col and clinic_col:
            tmp = df[[clinic_col, outcome_col]].copy()
            tmp[outcome_col] = tmp[outcome_col].astype(str)
            bad = tmp[outcome_col].str.contains(r"readmit|re[- ]?admit|bad|complication|fail", case=False, na=False)
            gp = (bad.groupby(tmp[clinic_col]).mean()*100).sort_values(ascending=False)
            return "No outcome/clinic data." if gp.empty else f"Highest adverse rate: {gp.index[0]} ({gp.iloc[0]:.1f}%)"
        return "Unsupported question or columns missing."

    if not q:
        st.text("Provide ?q=<exact title>."); st.stop()
    if q not in CATALOG:
        st.text("Question not in fixed catalog."); st.stop()

    ans = _compute_answer(q)
    st.text(ans); st.stop()

_api_mode_if_requested()
# ---------- END API MODE ----------

# ---------- Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file and hasattr(uploaded_file,"size"):
    mb = uploaded_file.size / (1024*1024)
    if mb > 100: st.warning(f"⚠️ Large file detected ({mb:.1f} MB). Some steps may take longer.")

# ... (everything below this point remains exactly the same as your original file)
