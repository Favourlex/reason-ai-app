# ================================================
# Reason AI — Universal Data Cleaner
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
# Call like: https://<app>/?api=1&question=Monthly%20revenue%20(sum%20of%20fee)
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

    # Build minimal helpers for API mode
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

    # Load df from session or latest artifacts
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

    result = {
        "ok": False,
        "error": None,
        "answer": None,
        "question": None,
        "supported_questions": [],
        "meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rows": int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0,
            "cols": int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0,
        }
    }

    q = None
    if isinstance(qp, dict):
        q = qp.get("question")
        if isinstance(q, list): q = q[0]
    result["question"] = q

    if df is None or df.empty:
        result["error"] = "No cleaned dataset available. Clean a CSV in the main app first."
        st.write(result); st.stop()

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

    result["supported_questions"] = CATALOG

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
        result["error"] = "Provide ?question=<exact title>. See supported_questions."
    else:
        if q not in CATALOG:
            result["error"] = "Question not in fixed catalog. Use exactly one of supported_questions."
        else:
            result["ok"] = True
            result["answer"] = _compute_answer(q)

    st.write(result)  # JSON-ish response
    st.stop()

_api_mode_if_requested()
# ---------- END API MODE ----------

# ---------- Upload ----------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file and hasattr(uploaded_file,"size"):
    mb = uploaded_file.size / (1024*1024)
    if mb > 100: st.warning(f"⚠️ Large file detected ({mb:.1f} MB). Some steps may take longer.")

if "df_clean" not in st.session_state: st.session_state["df_clean"] = None
if "clean_path" not in st.session_state: st.session_state["clean_path"] = None

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
            df.columns = [c.strip().replace(" ","_").replace("-","_").lower() for c in df.columns]

            missing_filled = 0
            for col in df.columns:
                if detect_numeric_series(df[col]):
                    nums = to_numeric_smart(df[col])
                    fill_val = nums.median()
                    missing_filled += nums.isna().sum()
                    df[col] = nums.fillna(fill_val)
                    continue
                if any(k in col for k in ["date","admission","discharge","created","visit"]):
                    df[col] = normalize_date_series(df[col]); continue
                if "gender" in col:
                    df[col] = df[col].astype(str).map(clean_gender_cell); continue
                # text default
                df[col] = df[col].astype(str).map(clean_text_cell)
                # opportunistic numeric rescue
                temp_num = to_numeric_smart(df[col])
                if temp_num.notna().sum()>0 and temp_num.isna().sum() < len(temp_num)*0.5:
                    median_val = temp_num.median()
                    missing_filled += temp_num.isna().sum()
                    df[col] = temp_num.fillna(median_val)
                else:
                    df[col] = df[col].apply(lambda x: x.title() if isinstance(x,str) and x.lower()!="unknown" else "Unknown")

            dup_count = df.duplicated().sum()
            df.drop_duplicates(inplace=True)

            status.update(label="✅ Cleaning complete", state="complete")

        # Summary
        st.subheader("📊 Data-Quality Summary")
        st.write(f"✅ **Rows retained:** {df.shape[0]}/{rows_before}")
        st.write(f"✅ **Columns retained:** {df.shape[1]}/{cols_before}")
        st.write(f"🧮 **Numeric columns detected:** {len(df.select_dtypes(include=np.number).columns)}")
        st.write(f"🔤 **Text columns detected:** {len(df.select_dtypes(exclude=np.number).columns)}")
        st.write(f"🧼 **Missing values filled:** {missing_filled}")
        st.write(f"🗑️ **Duplicates removed:** {dup_count}")

        # Preview + save
        st.subheader("🧾 Cleaned Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        os.makedirs("artifacts", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"artifacts/cleaned_{os.path.splitext(uploaded_file.name)[0]}_{ts}.csv"
        df.to_csv(clean_filename, index=False, encoding="utf-8")

        st.success(f"✅ Cleaned file saved successfully: `{clean_filename}`")
        st.session_state["df_clean"] = df
        st.session_state["clean_path"] = clean_filename

        st.download_button(
            label="⬇️ Download Cleaned Dataset",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"cleaned_{uploaded_file.name}",
            mime="text/csv"
        )

        report = f"""# Reason AI — Data Quality Report
- Rows retained: **{df.shape[0]}/{rows_before}**
- Columns retained: **{df.shape[1]}/{cols_before}**
- Numeric columns: **{len(df.select_dtypes(include=np.number).columns)}**
- Text columns: **{len(df.select_dtypes(exclude=np.number).columns)}**
- Missing values filled: **{missing_filled}**
- Duplicates removed: **{dup_count}**
- Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        st.download_button("⬇️ Download Data-Quality Report (MD)",
            data=report.encode("utf-8"), file_name="data_quality_report.md", mime="text/markdown")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ---------- NeuralSeek Console Upload (UNCHANGED) ----------
st.markdown("## 🔗 NeuralSeek Integration (Console Upload)")

CONSOLE_API_BASE = "https://consoleapi-usw.neuralseek.com"

def ns_console_upload(instance_id: str, api_key: str, file_path: str, file_mime="text/csv"):
    url = f"{CONSOLE_API_BASE}/{instance_id}/exploreUpload"
    headers = {"accept":"*/*","apikey":api_key}
    with open(file_path,"rb") as fh:
        files = {"file": (os.path.basename(file_path), fh, file_mime)}
        resp = requests.post(url, headers=headers, files=files, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Upload failed {resp.status_code}: {resp.text}")
    try: return resp.json()
    except ValueError:
        raise RuntimeError(f"Upload returned non-JSON. CT={resp.headers.get('Content-Type')} Body={resp.text[:300]}")

def ns_console_list_files(instance_id: str, api_key: str):
    url = f"{CONSOLE_API_BASE}/{instance_id}/exploreFiles"
    headers = {"accept":"application/json","apikey":api_key}
    resp = requests.get(url, headers=headers, timeout=60)
    if not (200 <= resp.status_code < 300):
        raise RuntimeError(f"/exploreFiles failed {resp.status_code} | CT:{resp.headers.get('Content-Type')} | Body:{resp.text[:300]}")
    text = (resp.text or "").strip()
    if not text: return []
    try: return resp.json()
    except ValueError: return []

with st.expander("Console API settings", expanded=True):
    ns_instance = st.text_input("NeuralSeek Instance ID", value="e4826687b7e8c357dfcd1b18")
    ns_api_key  = st.text_input("NeuralSeek **Console** API Key", type="password")
    debug_mode  = st.toggle("Show debug details", value=False)

colA, colB = st.columns([1,1])
with colA:
    push_btn = st.button("🚀 Send cleaned CSV to NeuralSeek")
with colB:
    if ns_instance.strip():
        st.link_button("🔗 Open mAIstro", f"https://console-usw.neuralseek.com/{ns_instance}/maistro#", use_container_width=True)
        st.link_button("🔗 Open Seek",    f"https://console-usw.neuralseek.com/{ns_instance}/seek#",    use_container_width=True)

if push_btn:
    if not st.session_state.get("clean_path"):
        st.error("Please clean and save a CSV first.")
    elif not ns_instance.strip() or not ns_api_key.strip():
        st.error("Provide both Instance ID and the **Console API** key.")
    else:
        try:
            with st.status("Uploading cleaned CSV to NeuralSeek Console API…", expanded=debug_mode) as s:
                if debug_mode: s.write(f"File: `{st.session_state['clean_path']}`")
                result = ns_console_upload(ns_instance.strip(), ns_api_key.strip(), st.session_state["clean_path"])
                uploaded_name = result.get("fn") or os.path.basename(st.session_state["clean_path"])
                if debug_mode: s.write(f"✅ Uploaded as: **{uploaded_name}**")

                ready = False
                for _ in range(20):
                    try:
                        files = ns_console_list_files(ns_instance.strip(), ns_api_key.strip())
                        if any(uploaded_name in str(x) for x in files):
                            ready = True; break
                    except Exception as diag:
                        if debug_mode: s.write(f"Polling note: {diag}")
                        break
                    time.sleep(2)

                if ready:
                    s.update(label="🎉 Ready in NeuralSeek → Use Document.", state="complete")
                    st.success("🎉 File is ready in mAIstro (Use Document → Local Document).")
                else:
                    s.update(label="✅ Upload complete.", state="complete")
                    st.success("✅ Upload complete. If not visible yet in mAIstro, wait a few seconds and click Send again.")
        except Exception as e:
            st.error(f"⚠️ NeuralSeek upload error: {e}")

# ---------- Local Q&A Showcase (FIXED questions; Ask vs Visualize) ----------
st.markdown("## 💬 Reason AI — Fixed Q&A (Ask first, then Visualize)")

df = st.session_state.get("df_clean")
if df is None or df.empty:
    st.info("Upload and clean a CSV first to enable Q&A.")
else:
    cols = list(df.columns)

    def find_first(*keys):
        keys_low = [k.lower() for k in keys]
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in keys_low): return c
        return None

    doctor_col  = find_first("doctor","physician","consultant")
    clinic_col  = find_first("clinic","location","ward","department")
    fee_col     = find_first("fee","amount","charge","bill","cost","price")
    date_col    = find_first("date","admission","discharge","visit","created")
    gender_col  = find_first("gender","sex")
    age_col     = find_first("age")
    proc_col    = find_first("procedure","treatment","service")
    outcome_col = find_first("outcome","readmit","readmission","status","result")

    # prepare fee numeric (fixed tiny typo)
    if fee_col is not None:
        try:
            df[fee_col] = to_numeric_smart(df[fee_col])
        except Exception:
            pass

    # month helper
    def month_series(c):
        try:
            d = pd.to_datetime(df[c].replace("Unknown", np.nan), errors="coerce")
            return d.dt.to_period("M").dropna().astype(str)
        except Exception:
            return pd.Series(dtype="object")

    # ---- Fixed Catalog of 10 questions (each with guards) ----
    # Each entry returns: (answer_text, chart_series_or_df, chart_type_hint)
    CATALOG = []

    if doctor_col:
        def q1():
            vc = df[doctor_col].replace("Unknown", np.nan).dropna().value_counts()
            ans = f"Total doctors: {vc.index.nunique()} | Top doctor: {vc.index[0]} ({int(vc.iloc[0])} records)" if not vc.empty else "No doctor data."
            return ans, vc.head(10), "bar"
        CATALOG.append(("Top doctors by patient count", q1))

    if clinic_col:
        def q2():
            vc = df[clinic_col].replace("Unknown", np.nan).dropna().value_counts()
            ans = f"Total clinics: {vc.index.nunique()} | Busiest clinic: {vc.index[0]} ({int(vc.iloc[0])} records)" if not vc.empty else "No clinic data."
            return ans, vc.head(10), "bar"
        CATALOG.append(("Top clinics by patient count", q2))

    if fee_col and doctor_col:
        def q3():
            tmp = df[[doctor_col, fee_col]].dropna()
            gp = tmp.groupby(doctor_col)[fee_col].mean().sort_values(ascending=False)
            ans = ("No fee/doctor data." if gp.empty
                   else f"Highest avg fee: {gp.index[0]} (₦{gp.iloc[0]:,.0f})")
            return ans, gp.head(8), "bar"
        CATALOG.append(("Average fee by doctor (Top 8)", q3))

    if fee_col and clinic_col:
        def q4():
            tmp = df[[clinic_col, fee_col]].dropna()
            gp = tmp.groupby(clinic_col)[fee_col].mean().sort_values(ascending=False)
            ans = ("No fee/clinic data." if gp.empty
                   else f"Highest avg fee clinic: {gp.index[0]} (₦{gp.iloc[0]:,.0f})")
            return ans, gp.head(8), "bar"
        CATALOG.append(("Average fee by clinic (Top 8)", q4))

    if date_col:
        def q5():
            m = month_series(date_col)
            s = m.value_counts().sort_index()
            ans = ("No date data." if s.empty else f"Monthly range: {s.index.min()} → {s.index.max()} | Peak: {s.idxmax()} ({int(s.max())})")
            return ans, s, "line"
        CATALOG.append(("Monthly case trend", q5))

    if date_col and fee_col:
        def q6():
            d = pd.to_datetime(df[date_col].replace("Unknown", np.nan), errors="coerce")
            tmp = pd.DataFrame({"month": d.dt.to_period("M").astype(str), "fee": df[fee_col]})
            tmp = tmp.dropna()
            gp = tmp.groupby("month")["fee"].sum().sort_index()
            ans = ("No date/fee data." if gp.empty
                   else f"Revenue window: {gp.index.min()} → {gp.index.max()} | Peak month: {gp.idxmax()} (₦{gp.max():,.0f})")
            return ans, gp, "line"
        CATALOG.append(("Monthly revenue (sum of fee)", q6))

    if gender_col:
        def q7():
            vc = df[gender_col].replace("Unknown", np.nan).dropna().value_counts()
            ans = ("No gender data." if vc.empty
                   else "Gender mix: " + ", ".join([f"{k}: {int(v)}" for k,v in vc.items()]))
            return ans, vc, "pie"
        CATALOG.append(("Gender distribution", q7))

    if age_col and detect_numeric_series(df[age_col]):
        def q8():
            s = to_numeric_smart(df[age_col]).dropna()
            ans = ("No age data." if s.empty
                   else f"Age — mean: {s.mean():.1f}, median: {s.median():.1f}, n={len(s)}")
            return ans, s, "hist"
        CATALOG.append(("Age distribution (histogram)", q8))

    if proc_col:
        def q9():
            vc = df[proc_col].replace("Unknown", np.nan).dropna().value_counts()
            ans = ("No procedure data." if vc.empty
                   else f"Top procedure: {vc.index[0]} ({int(vc.iloc[0])})")
            return ans, vc.head(10), "bar"
        CATALOG.append(("Top procedures by count", q9))

    if outcome_col and clinic_col:
        def q10():
            tmp = df[[clinic_col, outcome_col]].copy()
            tmp[outcome_col] = tmp[outcome_col].astype(str)
            bad = tmp[outcome_col].str.contains(r"readmit|re[- ]?admit|bad|complication|fail", case=False, na=False)
            gp = (bad.groupby(tmp[clinic_col]).mean()*100).sort_values(ascending=False)
            ans = ("No outcome/clinic data." if gp.empty
                   else f"Highest adverse rate: {gp.index[0]} ({gp.iloc[0]:.1f}%)")
            gp.name = "Rate (%)"
            return ans, gp.head(10), "bar"
        CATALOG.append(("Readmission/Adverse rate by clinic", q10))

    # If the dataset lacks some fields, you may end up with fewer than 10; that is intentional and safe.

    if not CATALOG:
        st.warning("This dataset doesn’t match the fixed question catalog. Rename columns (doctor/clinic/fee/date/gender/age/procedure/outcome) and re-upload.")
    else:
        titles = [t for t,_ in CATALOG]
        qcol1, qcol2 = st.columns([2,1])
        with qcol1:
            sel = st.selectbox("Choose a question (fixed set; other questions will be rejected):", titles, index=0)
        with qcol2:
            typed = st.text_input("Or type exactly (must match one of the fixed questions):", value=titles[0])

        ask_btn = st.button("🧩 Ask (text answer only)")
        viz_btn = st.button("📈 Get Visualization (from last answer)")

        # store last result for viz
        if "last_q_answer" not in st.session_state: st.session_state["last_q_answer"] = None
        if "last_q_data" not in st.session_state: st.session_state["last_q_data"] = None
        if "last_q_type" not in st.session_state: st.session_state["last_q_type"] = None
        if "last_q_title" not in st.session_state: st.session_state["last_q_title"] = None

        def run_question(title):
            fn = dict(CATALOG).get(title)
            if fn is None:
                st.error("Sorry, I can’t answer that question. It’s outside the approved dataset scope.")
                return
            ans, data, chart_kind = fn()
            st.session_state["last_q_answer"] = ans
            st.session_state["last_q_data"] = data
            st.session_state["last_q_type"] = chart_kind
            st.session_state["last_q_title"] = title
            st.success("Answer ready. You may now click **Get Visualization** if you want a chart.")
            st.markdown(f"**Answer:** {ans}")

        if ask_btn:
            # Only allow questions from the fixed set
            chosen = typed.strip() if typed.strip() in titles else sel
            if chosen not in titles:
                st.error("Sorry, I can’t answer that question. Please pick one from the list.")
            else:
                run_question(chosen)

        if viz_btn:
            data = st.session_state.get("last_q_data")
            chart_kind = st.session_state.get("last_q_type")
            title_hint = st.session_state.get("last_q_title") or "Visualization"
            if data is None:
                st.error("Run **Ask** first to compute the answer, then click **Get Visualization**.")
            else:
                # Render chart deterministically
                fig, ax = plt.subplots(figsize=(8,4))
                if chart_kind == "bar" and isinstance(data, pd.Series):
                    data.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Category"); ax.set_ylabel(data.name or "value")
                    plt.xticks(rotation=45, ha="right"); ax.set_title(title_hint); plt.tight_layout()
                elif chart_kind == "line" and isinstance(data, pd.Series):
                    data.plot(ax=ax)
                    ax.set_xlabel("Month"); ax.set_ylabel(data.name or "value")
                    ax.set_title(title_hint); plt.tight_layout()
                elif chart_kind == "pie" and isinstance(data, pd.Series):
                    ax.pie(data.values, labels=data.index, autopct="%1.1f%%", startangle=140)
                    ax.set_title(title_hint); plt.tight_layout()
                elif chart_kind == "hist" and isinstance(data, pd.Series):
                    data.dropna().plot(kind="hist", bins=20, ax=ax)
                    ax.set_xlabel(data.name or "value"); ax.set_title(title_hint); plt.tight_layout()
                else:
                    plt.close(fig)
                    st.info("No chart available for this answer.")
                if plt.get_fignums():
                    st.pyplot(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; font-size:15px; color:gray; margin-top:10px;'>"
    "💡 Developed by <b>Favour Ezeofor</b>"
    "</div>",
    unsafe_allow_html=True
)
