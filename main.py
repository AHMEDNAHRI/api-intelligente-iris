import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List

import joblib
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

app = FastAPI(title="API Iris", version="2.0")
model = joblib.load("iris_model.pkl")
NOMS = ["setosa", "versicolor", "virginica"]
DB = "predictions.db"

# ---------- Limitation de debit ----------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    lambda request, exc: JSONResponse(
        status_code=429, content={"detail": "Rate limit exceeded"}
    ),
)

# ---------- Authentification (API Key) + roles ----------
API_KEYS = {
    "user-key-123": {"username": "alice", "role": "USER"},
    "admin-key-456": {"username": "admin", "role": "ADMIN"},
}
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_current_user(key: str = Security(api_key_header)):
    if key is None or key not in API_KEYS:
        raise HTTPException(status_code=401, detail="API Key manquante ou invalide")
    return API_KEYS[key]


def require_admin(user: dict = Depends(get_current_user)):
    if user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Acces reserve aux ADMIN")
    return user


# ---------- Base SQLite ----------
def db_execute(sql, params=()):
    conn = sqlite3.connect(DB)
    try:
        rows = conn.execute(sql, params).fetchall()
        conn.commit()
        return rows
    finally:
        conn.close()


db_execute(
    "CREATE TABLE IF NOT EXISTS predictions ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT, timestamp TEXT, "
    "username TEXT, input TEXT, prediction TEXT, confidence REAL)"
)


def save_prediction(request_id, ts, username, flower, res):
    db_execute(
        "INSERT INTO predictions (request_id, timestamp, username, input, "
        "prediction, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        (request_id, ts, username, json.dumps(flower.model_dump()),
         res["prediction"], res["confidence"]),
    )


# ---------- Schemas ----------
class IrisInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


class IrisBatch(BaseModel):
    flowers: List[IrisInput] = Field(min_length=1, max_length=100)


def predict_one(f: IrisInput):
    x = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
    proba = model.predict_proba(x)[0]
    idx = int(proba.argmax())
    return {
        "prediction": NOMS[idx],
        "confidence": round(float(proba[idx]), 2),
        "probabilities": {NOMS[i]: round(float(proba[i]), 3) for i in range(3)},
    }


def new_meta():
    return str(uuid.uuid4()), datetime.now(timezone.utc).isoformat()


# ---------- Endpoints ----------
@app.get("/")
def home():
    return {"message": "Bienvenue dans l'API Iris"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisInput, user: dict = Depends(get_current_user)):
    request_id, ts = new_meta()
    res = predict_one(data)
    save_prediction(request_id, ts, user["username"], data, res)
    return {"request_id": request_id, "timestamp": ts, **res, "status": "success"}


@app.post("/predict/batch")
def predict_batch(batch: IrisBatch, user: dict = Depends(get_current_user)):
    request_id, ts = new_meta()
    results = []
    for f in batch.flowers:
        res = predict_one(f)
        save_prediction(request_id, ts, user["username"], f, res)
        results.append(res)
    return {"request_id": request_id, "timestamp": ts,
            "results": results, "status": "success"}


@app.get("/history")
def history(user: dict = Depends(get_current_user)):
    cols = ["request_id", "timestamp", "username", "prediction", "confidence"]
    base = "SELECT request_id, timestamp, username, prediction, confidence FROM predictions"
    if user["role"] == "ADMIN":
        rows = db_execute(base + " ORDER BY id DESC LIMIT 50")
    else:
        rows = db_execute(base + " WHERE username = ? ORDER BY id DESC LIMIT 50",
                          (user["username"],))
    return [dict(zip(cols, r)) for r in rows]


@app.get("/admin/stats")
def stats(user: dict = Depends(require_admin)):
    total = db_execute("SELECT COUNT(*) FROM predictions")[0][0]
    return {"total_predictions": total}


@app.get("/limited")
@limiter.limit("5/minute")
def limited(request: Request):
    return {"message": "Requete autorisee"}
