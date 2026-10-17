"""Backend API Developer + QA/Observability: ONNX REST API with logging and drift alerts.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Env:  API_KEY (optional, enables X-API-Key auth), MODEL_DIR, LATENCY_SLA_MS,
      DRIFT_WINDOW, DRIFT_Z
"""
import json
import logging
import math
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from features import FEATURES, msg_features, normalize

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
API_KEY = os.getenv("API_KEY")                      # secret via env var, never in git
LATENCY_SLA_MS = float(os.getenv("LATENCY_SLA_MS", "100"))
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "50"))  # requests per drift check
DRIFT_Z = float(os.getenv("DRIFT_Z", "4.0"))         # |z| threshold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("spam-api")

app = FastAPI(title="SMS Spam Detector", version="1.0")
session = ort.InferenceSession(str(MODEL_DIR / "spam_model.onnx"),
                               providers=["CPUExecutionProvider"])
reference = json.loads((MODEL_DIR / "reference_stats.json").read_text())
window = deque(maxlen=DRIFT_WINDOW)
state = {"drift_alert": False, "drift_detail": {}}


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)


class PredictResponse(BaseModel):
    label: str
    spam_probability: float
    latency_ms: float
    drift_alert: bool


def check_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid API key")


def check_drift():
    """z-test of the window mean of each input feature vs the training reference."""
    if len(window) < DRIFT_WINDOW:
        return
    detail = {}
    for f in FEATURES:
        ref = reference[f]
        mean = sum(w[f] for w in window) / len(window)
        se = max(ref["std"], 1e-9) / math.sqrt(len(window))
        detail[f] = round((mean - ref["mean"]) / se, 2)
    alert = any(abs(z) > DRIFT_Z for z in detail.values())
    state.update(drift_alert=alert, drift_detail=detail)
    if alert:
        log.warning("DRIFT ALERT z-scores=%s", detail)
    window.clear()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/drift")
def drift():
    return state


@app.post("/v1/predict", response_model=PredictResponse, dependencies=[Depends(check_key)])
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    text = normalize(req.message)
    if not text:
        raise HTTPException(status_code=422, detail="message is empty after cleaning")
    label_idx, probs = session.run(None, {"message": np.array([[text]], dtype=object)})
    spam_p = float(probs[0][1])
    window.append(msg_features(req.message))
    check_drift()
    latency = (time.perf_counter() - t0) * 1000
    log.info("predict label=%s p_spam=%.3f latency_ms=%.2f len=%d",
             "spam" if label_idx[0] == 1 else "ham", spam_p, latency, len(req.message))
    if latency > LATENCY_SLA_MS:
        log.warning("LATENCY SLA breach: %.1f ms > %.0f ms", latency, LATENCY_SLA_MS)
    return PredictResponse(label="spam" if label_idx[0] == 1 else "ham",
                           spam_probability=round(spam_p, 4),
                           latency_ms=round(latency, 2),
                           drift_alert=state["drift_alert"])
