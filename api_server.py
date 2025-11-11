# ===========================================================
# Reason AI Backend (API) – Data Hosting + Query Endpoint
# ===========================================================
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd, os

app = FastAPI(title="Reason AI API")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
LATEST_CSV = os.path.join(DATA_DIR, "cleaned_latest.csv")

@app.get("/")
def root():
    return {"ok": True, "message": "Reason AI Backend is Live 👑"}

@app.post("/upload_cleaned")
async def upload_cleaned(file: UploadFile):
    try:
        contents = await file.read()
        with open(LATEST_CSV, "wb") as f:
            f.write(contents)
        return {"ok": True, "message": "Cleaned dataset uploaded successfully."}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/data/latest")
def get_latest_data():
    if not os.path.exists(LATEST_CSV):
        return JSONResponse({"ok": False, "error": "No cleaned dataset uploaded yet."}, status_code=404)
    return FileResponse(LATEST_CSV, media_type="text/csv")

@app.get("/query")
def query(q: str):
    if not os.path.exists(LATEST_CSV):
        return {"ok": False, "error": "No cleaned dataset available."}
    df = pd.read_csv(LATEST_CSV)
    ql = q.lower()
    if "doctor" in ql:
        counts = df[df.columns[df.columns.str.contains("doctor", case=False)][0]].value_counts()
        return {"ok": True, "response": {"Top doctor": counts.idxmax(), "Counts": counts.to_dict()}}
    elif "clinic" in ql:
        counts = df[df.columns[df.columns.str.contains("clinic|hospital", case=False)][0]].value_counts()
        return {"ok": True, "response": {"Top clinic": counts.idxmax(), "Counts": counts.to_dict()}}
    elif "gender" in ql:
        counts = df[df.columns[df.columns.str.contains("gender|sex", case=False)][0]].value_counts()
        return {"ok": True, "response": {"Gender distribution": counts.to_dict()}}
    else:
        return {"ok": True, "response": f"Query '{q}' received, but no direct logic matches yet."}
