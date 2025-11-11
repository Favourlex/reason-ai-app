# ===========================================================
# Reason AI — Universal Data Cleaner + Fixed Q&A + NeuralSeek Integration
# with Navigation, Dashboard & Settings
# Developed by Favour Ezeofor 👑
# ===========================================================

import streamlit as st
import pandas as pd
import numpy as np
import os, time, requests
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="Reason AI – Universal Data Cleaner",
                   layout="wide", page_icon="🧠")

# ----------------- HEADER -----------------
st.markdown("""
<h1 style="font-size:40px; font-weight:700; text-align:center;">🧠 Reason AI</h1>
<h3 style="text-align:center;">Universal Data Cleaning & Analysis System</h3>
<p style="text-align:center;">Developed by <b>Favour Ezeofor 👑</b></p>
""", unsafe_allow_html=True)
st.markdown("---")

# ----------------- NAVIGATION -----------------
nav = st.radio("Navigate",
               ["🧹 Data Cleaner", "📊 Dashboard", "⚙️ Settings"],
               horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ----------------- HELPERS -----------------
NULL_TOKENS = {"", " ", "nan", "none", "null", "n/a", "na", "unknown"}
GENDER_MAP = {
    "m":"Male","male":"Male","man":"Male","boy":"Male",
    "f":"Female","female":"Female","woman":"Female","girl":"Female",
    "others":"Other","other":"Other","non-binary":"Other","nb":"Other"
}

def clean_text_cell(x):
    if pd.isna(x): return "Unknown"
    s = str(x).strip()
    return "Unknown" if s.lower() in NULL_TOKENS else s.title()

def to_numeric_smart(s):
    s = s.astype(str).str.replace(r"[₦$,()% ]","",regex=True)
    return pd.to_numeric(s, errors="coerce")

def normalize_date_series(s):
    d = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    return d.dt.strftime("%Y-%m-%d").fillna("Unknown")

def clean_gender_cell(x):
    s = str(x).strip().lower()
    if s in NULL_TOKENS: return "Unknown"
    return GENDER_MAP.get(s, "Unknown")

@st.cache_data(show_spinner=False)
def read_csv(file_bytes):
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try: return pd.read_csv(BytesIO(file_bytes), encoding=enc), enc
        except UnicodeDecodeError: continue
    return pd.read_csv(BytesIO(file_bytes)), "auto"

# ===========================================================
# 1️⃣ DATA CLEANER TAB
# ===========================================================
if nav.startswith("🧹"):
    st.markdown("""
    Upload any **CSV file** and Reason AI will automatically clean it:
    - Remove empty rows/columns  
    - Handle missing values  
    - Detect and normalize dates  
    - Standardize gender markers  
    - Output a ready CSV for NeuralSeek ingestion  
    """)

    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

    if uploaded:
        df, enc = read_csv(uploaded.getvalue())
        st.success(f"✅ Loaded **{uploaded.name}** (encoding: {enc})")
        rows0, cols0 = df.shape

        with st.status("🧽 Cleaning in progress..."):
            df.dropna(how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)
            df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

            for c in df.columns:
                if "gender" in c:
                    df[c] = df[c].astype(str).map(clean_gender_cell)
                elif "date" in c:
                    df[c] = normalize_date_series(df[c])
                else:
                    df[c] = df[c].astype(str).map(clean_text_cell)
                    num = to_numeric_smart(df[c])
                    if num.notna().sum() > len(num)*0.5:
                        df[c] = num.fillna(num.median())

        st.success("🎉 Cleaning complete!")
        st.dataframe(df.head(10), use_container_width=True)

        os.makedirs("artifacts", exist_ok=True)
        path = f"artifacts/cleaned_{os.path.splitext(uploaded.name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(path, index=False)
        df.to_csv("artifacts/cleaned_latest.csv", index=False)
        st.session_state["df_clean"] = df
        st.session_state["clean_path"] = path

        st.download_button("⬇️ Download Cleaned Dataset",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"cleaned_{uploaded.name}",
                           mime="text/csv")

        st.info("✅ Cleaned file saved and `cleaned_latest.csv` updated for API backend.")

        # ---------- NeuralSeek Console Upload ----------
        st.markdown("### 🔗 NeuralSeek Console Integration")
        inst = st.text_input("NeuralSeek Instance ID",
                             value=st.session_state.get("instance_id","e4826687b7e8c357dfcd1b18"))
        key  = st.text_input("NeuralSeek API Key",
                             type="password",
                             value=st.session_state.get("api_key",""))

        if st.button("🚀 Send Cleaned CSV to NeuralSeek"):
            try:
                url = f"https://consoleapi-usw.neuralseek.com/{inst}/exploreUpload"
                headers = {"accept":"*/*","apikey":key}
                with open(st.session_state["clean_path"],"rb") as f:
                    files = {"file":(os.path.basename(st.session_state["clean_path"]),f,"text/csv")}
                    r = requests.post(url, headers=headers, files=files, timeout=120)
                r.raise_for_status()
                st.success("✅ Uploaded successfully — check mAIstro → Use Document.")
            except Exception as e:
                st.error(f"⚠️ Upload failed: {e}")

        st.link_button("🔗 Open mAIstro", f"https://console-usw.neuralseek.com/{inst}/maistro#")
        st.link_button("🔗 Open Seek", f"https://console-usw.neuralseek.com/{inst}/seek#")


# ===========================================================
# 2️⃣ DASHBOARD (Q&A + VISUALIZATION)
# ===========================================================
elif nav.startswith("📊"):
    df = st.session_state.get("df_clean")
    if df is None or df.empty:
        st.info("📁 Upload and clean a CSV first to enable Q&A.")
    else:
        st.markdown("## 💬 Reason AI — Fixed Q&A (Ask & Visualize)")
        cols = list(df.columns)

        def find_first(*keys):
            for c in cols:
                if any(k in c.lower() for k in keys):
                    return c
            return None

        doctor = find_first("doctor","physician","consultant")
        clinic = find_first("clinic","ward","department")
        fee = find_first("fee","amount","charge","bill")
        date = find_first("date","admission","discharge","visit")
        gender = find_first("gender","sex")
        age = find_first("age")
        proc = find_first("procedure","treatment")
        outcome = find_first("outcome","status","result")

        # Ensure fee numeric
        if fee: df[fee] = to_numeric_smart(df[fee])

        CATALOG = []
        if doctor: CATALOG.append(("Top doctors by patient count",
            lambda: (f"Top doctor: {df[doctor].value_counts().idxmax()}", df[doctor].value_counts(), "bar")))
        if clinic: CATALOG.append(("Top clinics by patient count",
            lambda: (f"Top clinic: {df[clinic].value_counts().idxmax()}", df[clinic].value_counts(), "bar")))
        if fee and doctor: CATALOG.append(("Average fee by doctor (Top 8)",
            lambda: (f"Highest avg fee: {df.groupby(doctor)[fee].mean().idxmax()}", df.groupby(doctor)[fee].mean().sort_values(ascending=False).head(8), "bar")))
        if fee and clinic: CATALOG.append(("Average fee by clinic (Top 8)",
            lambda: (f"Highest avg clinic: {df.groupby(clinic)[fee].mean().idxmax()}", df.groupby(clinic)[fee].mean().sort_values(ascending=False).head(8), "bar")))
        if date and fee: CATALOG.append(("Monthly revenue (sum of fee)",
            lambda: (f"Peak month revenue: {pd.to_datetime(df[date],errors='coerce').dt.to_period('M').value_counts().idxmax()}", df.groupby(pd.to_datetime(df[date],errors='coerce').dt.to_period('M'))[fee].sum(), "line")))
        if gender: CATALOG.append(("Gender distribution",
            lambda: ("Gender distribution", df[gender].value_counts(), "pie")))

        titles = [t for t,_ in CATALOG]
        sel = st.selectbox("Choose a question:", titles)
        ask = st.button("🧩 Ask Question")
        viz = st.button("📈 Visualize Answer")

        if ask:
            ans, data, kind = dict(CATALOG)[sel]()
            st.session_state["ans"]=ans;st.session_state["data"]=data;st.session_state["kind"]=kind
            st.success(f"**Answer:** {ans}")

        if viz and "data" in st.session_state:
            data=st.session_state["data"]; kind=st.session_state["kind"]; title=sel
            fig, ax = plt.subplots(figsize=(8,4))
            if kind=="bar": data.plot(kind="bar", ax=ax)
            elif kind=="line": data.plot(ax=ax)
            elif kind=="pie": ax.pie(data.values, labels=data.index, autopct="%1.1f%%")
            st.pyplot(fig, use_container_width=True)


# ===========================================================
# 3️⃣ SETTINGS TAB
# ===========================================================
else:
    st.markdown("### ⚙️ App Settings")
    instance_id = st.text_input("Default NeuralSeek Instance ID",
                                value=st.session_state.get("instance_id",""))
    api_key = st.text_input("Default NeuralSeek API Key",
                            type="password",
                            value=st.session_state.get("api_key",""))
    if st.button("💾 Save Settings"):
        st.session_state["instance_id"]=instance_id
        st.session_state["api_key"]=api_key
        st.success("✅ Settings saved for this session.")

st.markdown("<hr><center>💡 Developed by <b>Favour Ezeofor</b> — Reason AI</center>",
            unsafe_allow_html=True)
