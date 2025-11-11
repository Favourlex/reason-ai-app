# ================================================================
#  Reason AI - Backend API Service (FastAPI)
#  Connects the Streamlit Data Cleaner with NeuralSeek REST nodes
#  Developed by Favour Ezeofor | Maintained by Anim (Crown Dev)
# ================================================================

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import requests
import io

# ------------------------------------------------------------
# 🔗 1. Initialize FastAPI app
# ------------------------------------------------------------
app = FastAPI(
    title="Reason AI Backend",
    description="FastAPI service for Reason AI – connects cleaned Streamlit dataset to NeuralSeek.",
    version="2.0.0"
)

# ------------------------------------------------------------
# 📂 2. Streamlit app data source
# ------------------------------------------------------------
# This is the public link to your Reason AI Streamlit app
STREAMLIT_DATA_URL = "https://reason-ai-app-u6jlsdnvxpz8u5yvsvittx.streamlit.app/artifacts/cleaned_latest.csv"

# ------------------------------------------------------------
# 🧩 3. Utility to load dataset dynamically
# ------------------------------------------------------------
def load_dataset():
    try:
        response = requests.get(STREAMLIT_DATA_URL, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except Exception as e:
        print("⚠️ Error loading dataset:", e)
        return None

# ------------------------------------------------------------
# 💬 4. Root endpoint
# ------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "active",
        "message": "✅ Reason AI API backend is live and ready.",
        "usage": {
            "example_query": "/query?q=Top%20doctors%20by%20patient%20count",
            "data_source": STREAMLIT_DATA_URL
        }
    }

# ------------------------------------------------------------
# 🔍 5. Query endpoint for NeuralSeek and REST node
# ------------------------------------------------------------
@app.get("/query")
def query_data(q: str = Query(..., description="Natural language query about the dataset")):
    df = load_dataset()
    if df is None:
        return JSONResponse(
            status_code=404,
            content={"ok": False, "error": "No cleaned dataset available. Please run Streamlit app first."}
        )

    q_lower = q.lower()
    result = {}

    try:
        # Example intent recognition
        if "doctor" in q_lower and "count" in q_lower:
            top_doctors = df["doctor_name"].value_counts().head(5).to_dict()
            result = {"query_type": "top_doctors_by_patient_count", "data": top_doctors}

        elif "city" in q_lower or "location" in q_lower:
            city_counts = df["city"].value_counts().head(5).to_dict()
            result = {"query_type": "patients_by_city", "data": city_counts}

        elif "disease" in q_lower:
            disease_stats = df["disease"].value_counts().head(5).to_dict()
            result = {"query_type": "common_diseases", "data": disease_stats}

        else:
            result = {
                "query_type": "dataset_summary",
                "rows": len(df),
                "columns": list(df.columns)
            }

        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "source": "Reason AI Streamlit dataset",
                "response": result
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"Query processing failed: {str(e)}"}
        )

# ------------------------------------------------------------
# 🧠 6. Health Check
# ------------------------------------------------------------
@app.get("/health")
def health_check():
    df = load_dataset()
    return {
        "status": "live",
        "data_available": df is not None,
        "message": "✅ Backend is running and ready for NeuralSeek integration."
    }

# ------------------------------------------------------------
# ⚙️ 7. Server start command (Render uses this automatically)
# ------------------------------------------------------------
# Command used on Render:
# uvicorn api_server:app --host 0.0.0.0 --port 10000
# ------------------------------------------------------------
