# ================================================
# 🧠 Reason AI — Universal Data Cleaner
# + NeuralSeek Console Upload
# + Fixed Q&A Visualization
# + Exports cleaned_latest.csv for FastAPI backend
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
    "m": "Male", "male": "Male", "man": "Male", "boy": "Male",
    "f": "Female", "female": "Female", "woman": "Female", "girl": "Female",
    "others": "Other", "other": "Other", "non-binary": "Other", "nonbinary": "Other",
    "nb": "Other", "trans": "Other", "transgender": "Other"
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
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(BytesIO(file_bytes), encoding=enc), enc
        except UnicodeDecodeError:
            continue
    return pd.read_csv(BytesIO(file_bytes)), "auto"

# ---------- Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file and hasattr(uploaded_file, "size"):
    mb = uploaded_file.size / (1024*1024)
    if mb > 100:
        st.warning(f"⚠️ Large file detected ({mb:.1f} MB). Some steps may take longer.")

if "df_clean" not in st.session_state: st.session_state["df_clean"] = None

# ---------- Clean ----------
if uploaded_file:
    try:
        file_bytes = uploaded_file.getvalue()
        df, enc_used = _read_csv_cached(file_bytes, uploaded_file.name)
        st.caption(f"📥 Loaded **{uploaded_file.name}** using encoding **{enc_used}**")
        st.success("✅ File uploaded successfully!")

        rows_before, cols_before = df.shape

        with st.status("🧽 Cleaning in progress...", expanded=False) as status:
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            df.columns = [c.strip().replace(" ", "_").replace("-", "_").lower() for c in df.columns]

            missing_filled = 0
            for col in df.columns:
                if detect_numeric_series(df[col]):
                    nums = to_numeric_smart(df[col])
                    fill_val = nums.median()
                    missing_filled += nums.isna().sum()
                    df[col] = nums.fillna(fill_val)
                    continue
                if any(k in col for k in ["date", "admission", "discharge", "created", "visit"]):
                    df[col] = normalize_date_series(df[col]); continue
                if "gender" in col:
                    df[col] = df[col].astype(str).map(clean_gender_cell); continue
                df[col] = df[col].astype(str).map(clean_text_cell)
                temp_num = to_numeric_smart(df[col])
                if temp_num.notna().sum() > 0 and temp_num.isna().sum() < len(temp_num)*0.5:
                    median_val = temp_num.median()
                    missing_filled += temp_num.isna().sum()
                    df[col] = temp_num.fillna(median_val)
                else:
                    df[col] = df[col].apply(lambda x: x.title() if isinstance(x, str) and x.lower() != "unknown" else "Unknown")

            dup_count = df.duplicated().sum()
            df.drop_duplicates(inplace=True)

            status.update(label="✅ Cleaning complete", state="complete")

        st.session_state["df_clean"] = df
        os.makedirs("artifacts", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_path = f"artifacts/cleaned_{os.path.splitext(uploaded_file.name)[0]}_{ts}.csv"
        df.to_csv(clean_path, index=False, encoding="utf-8")

        # Save a permanent copy for FastAPI backend
        df.to_csv("artifacts/cleaned_latest.csv", index=False, encoding="utf-8")

        st.success(f"✅ Cleaned file saved: `{clean_path}` and updated `cleaned_latest.csv` for API use.")

        st.download_button(
            label="⬇️ Download Cleaned Dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"cleaned_{uploaded_file.name}",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:15px; color:gray;'>"
    "💡 Developed by <b>Favour Ezeofor</b> — Reason AI"
    "</div>", unsafe_allow_html=True
)
