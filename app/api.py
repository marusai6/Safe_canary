import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.model import Model

app = FastAPI()

# Загружаем модель при старте
MODEL_VERSION = os.getenv("MODEL_VERSION", "unknown")
MODEL_FILE = os.getenv("MODEL_FILE", "")
if not MODEL_FILE:
    raise RuntimeError("MODEL_FILE env var is required")

model = Model(MODEL_FILE, MODEL_VERSION)

class PredictRequest(BaseModel):
    x1: float
    x2: float

class PredictResponse(BaseModel):
    prediction: int
    model_version: str
    request_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"status": "ready"}

@app.get("/version")
def version():
    return {"model_version": model.version}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        label, _ = model.predict(req.x1, req.x2)
        return PredictResponse(
            prediction=label,
            model_version=model.version,
            request_id="dummy-req-id", # В реальности здесь UUID
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
