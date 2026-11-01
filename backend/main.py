import json
import os
from pathlib import Path
from typing import Annotated

import numpy as np
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data_loader import window_features  # same function the model was trained with

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
booster = xgb.Booster()
booster.load_model(str(MODEL_DIR / "xgb.json"))
meta = json.loads((MODEL_DIR / "meta.json").read_text())
FEATURES = meta["features"]

app = FastAPI(title="IOH early warning (simulated-data demo)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    readings: list[Annotated[float, Field(ge=20, le=200)]] = Field(min_length=10, max_length=10)
    age: float = Field(ge=1, le=110)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    feats = window_features(np.array(req.readings, dtype=float), req.age)
    dm = xgb.DMatrix(np.array([[feats[f] for f in FEATURES]]), feature_names=FEATURES)
    risk = float(booster.predict(dm)[0])
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]  # last value is the bias
    top = sorted(zip(FEATURES, contribs), key=lambda t: -abs(t[1]))[:5]
    return {
        "risk": risk,
        "alert": risk >= meta["threshold"],
        "threshold": meta["threshold"],
        "drivers": [{"feature": f, "value": float(feats[f]), "push": float(c)} for f, c in top],
    }