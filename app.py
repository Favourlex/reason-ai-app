# ===========================================================
# Reason AI — Universal Data Cleaner + NeuralSeek Integration
# Streamlit Frontend (with Render Backend Bridge)
# Developed by Favour Ezeofor 👑
# ===========================================================

import streamlit as st, pandas as pd, numpy as np, os, requests
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

st.set_page_config(page_title="Reason AI – Universal Data Cleaner", layout="wide", page_icon="🧠")

# ----------------- HEADER -----------------
st.markdown("""
<h1 style="text-align:center;">🧠 Reason AI</h1>
<h3 style="text-align:center;">Universal Data Cleaning & Analysis System</h3>
<p style="text-align:center;">Developed by <b>Favour Ezeofor 👑</b></p>
""", unsafe_allow_html=True)
st.markdown("---")

# ----------------- NAVIGATION -----------------
nav = st.radio("Navigate", ["🧹 Data Cleaner", "📊 Dashboard", "⚙️ Settings"],
               horizontal=True, label_visibility="collapsed")
st.markdown("---")

# ----------------- HELPERS -----------------
NULL_TOKENS = {"", " ", "nan", "none", "null", "n/a", "na", "unknown"}
GENDER_MAP = {"m":"Male","male":"Male","man":"Male","boy":"Male",
              "f":"Female","female":"Female","woman":"Female","girl":"Female",
              "others":"Other","other":"Other","non-binary":"Other","nb":"Other"}

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
# 1️⃣ DATA CLEANER
# ===========================================================
if nav.startswith("🧹"):
    st.markdown("""
    Upload any CSV file — Reason AI will automatically clean it:
    - Remove empty rows/columns  
    - Handle missing values  
    - Detect and normalize dates  
    - Standardize gender markers  
    - Output a ready CSV for NeuralSeek and Render backend  
    """)

    uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])
    if uploaded:
        df, enc = read_csv(uploaded.getvalue())
        st.success(f"✅ Loaded **{uploaded.name}** (encoding: {enc})")

        with st.status("🧽 Cleaning in progress..."):
            df.dropna(how="all", inplace=True)
            df.dropna(axis=1, how="all", inplace=True)
            df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

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
        clean_path = f"artifacts/cleaned_{os.path.splitext(uploaded.name)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(clean_path, index=False)
        df.to_csv("artifacts/cleaned_latest.csv", index=False)

        st.session_state["df_clean"] = df
        st.session_state["clean_path"] = clean_path
        st.download_button("⬇️ Download Cleaned Dataset",
                           data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"cleaned_{uploaded.name}", mime="text/csv")

        # ---------- UPLOAD TO RENDER BACKEND ----------
        st.markdown("### 🌐 Upload to Reason AI Backend (for NeuralSeek & API)")
        BACKEND_URL = "https://reason-ai-api.onrender.com/upload_cleaned"
        if st.button("🚀 Sync with Backend"):
            try:
                with open(clean_path, "rb") as f:
                    r = requests.post(BACKEND_URL, files={"file":f})
                if r.status_code == 200:
                    st.success("✅ Uploaded to backend successfully!")
                    st.markdown(f"Backend data URL: `https://reason-ai-api.onrender.com/data/latest`")
                else:
                    st.error(f"⚠️ Upload failed: {r.text}")
            except Exception as e:
                st.error(f"⚠️ Error uploading: {e}")

        # ---------- NeuralSeek Console Integration ----------
        st.markdown("### 🤖 NeuralSeek Console Integration")
        inst = st.text_input("NeuralSeek Instance ID", "e4826687b7e8c357dfcd1b18")
        key  = st.text_input("NeuralSeek API Key", type="password")
        if st.button("📤 Send Cleaned CSV to NeuralSeek"):
            try:
                url = f"https://consoleapi-usw.neuralseek.com/{inst}/exploreUpload"
                headers = {"accept":"*/*","apikey":key}
                with open(clean_path, "rb") as f:
                    r = requests.post(url, headers=headers, files={"file":(os.path.basename(clean_path),f,"text/csv")})
                r.raise_for_status()
                st.success("✅ Uploaded to NeuralSeek successfully!")
            except Exception as e:
                st.error(f"⚠️ Upload failed: {e}")

        st.link_button("🔗 Open mAIstro", f"https://console-usw.neuralseek.com/{inst}/maistro#")
        st.link_button("🔗 Open Seek", f"https://console-usw.neuralseek.com/{inst}/seek#")

# ===========================================================
# 2️⃣ DASHBOARD
# ===========================================================
elif nav.startswith("📊"):
    df = st.session_state.get("df_clean")
    if df is None or df.empty:
        st.info("📁 Upload and clean a CSV first to enable Q&A.")
    else:
        st.markdown("## 💬 Reason AI – Fixed Q&A (Ask & Visualize)")
        cols = list(df.columns)

        def find_first(*keys):
            for c in cols:
                if any(k in c.lower() for k in keys): return c
            return None

        doctor=find_first("doctor","physician"); clinic=find_first("clinic","hospital")
        fee=find_first("fee","amount","bill"); date=find_first("date")
        gender=find_first("gender","sex")

        if fee: df[fee] = to_numeric_smart(df[fee])
        CATALOG=[]
        if doctor: CATALOG.append(("Top doctors by patient count",
            lambda:(f"Top doctor: {df[doctor].value_counts().idxmax()}", df[doctor].value_counts(),"bar")))
        if clinic: CATALOG.append(("Top clinics by patient count",
            lambda:(f"Top clinic: {df[clinic].value_counts().idxmax()}", df[clinic].value_counts(),"bar")))
        if gender: CATALOG.append(("Gender distribution",
            lambda:("Gender distribution", df[gender].value_counts(),"pie")))

        sel=st.selectbox("Choose a question:",[t for t,_ in CATALOG])
        if st.button("🧩 Ask"):
            ans,data,kind=dict(CATALOG)[sel]()
            st.session_state.update({"ans":ans,"data":data,"kind":kind})
            st.success(f"**Answer:** {ans}")
        if "data" in st.session_state:
            data,kind=st.session_state["data"],st.session_state["kind"]
            fig,ax=plt.subplots(figsize=(7,4))
            if kind=="bar": data.plot(kind="bar",ax=ax)
            elif kind=="pie": ax.pie(data.values,labels=data.index,autopct="%1.1f%%")
            st.pyplot(fig)

# ===========================================================
# 3️⃣ SETTINGS
# ===========================================================
else:
    st.markdown("### ⚙️ Settings")
    inst = st.text_input("Default NeuralSeek Instance ID", value=st.session_state.get("instance_id",""))
    key = st.text_input("Default NeuralSeek API Key", type="password", value=st.session_state.get("api_key",""))
    if st.button("💾 Save"):
        st.session_state["instance_id"]=inst
        st.session_state["api_key"]=key
        st.success("✅ Settings saved.")
st.markdown("<hr><center>💡 Developed by <b>Favour Ezeofor</b> — Reason AI</center>", unsafe_allow_html=True)
