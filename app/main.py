import json
import logging
import os
import time
import uuid

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse

logging.basicConfig(level=logging.INFO,
                    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("defect-api")

MODEL_DIR = os.getenv("MODEL_DIR", "models")
MAX_BYTES = int(os.getenv("MAX_BYTES", 5 * 1024 * 1024))   # upload size limit
MAX_SIDE = int(os.getenv("MAX_SIDE", 4096))                # reject absurdly large images
ALLOWED = {"image/jpeg", "image/png"}

# Loaded once at startup, not per request.
meta = json.load(open(f"{MODEL_DIR}/metadata.json"))
session = ort.InferenceSession(f"{MODEL_DIR}/model.onnx", providers=["CPUExecutionProvider"])
MEAN = np.array(meta["mean"], dtype=np.float32)
STD = np.array(meta["std"], dtype=np.float32)

app = FastAPI(title="Casting Defect Detection API", version=meta["version"])


def preprocess(img_bgr):
    """Must match training exactly: OpenCV decode, RGB, bilinear resize, ImageNet normalisation."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (meta["size"], meta["size"]), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def infer(img_bgr):
    t0 = time.perf_counter()
    logits = session.run(None, {"image": preprocess(img_bgr)})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float(probs[1]), (time.perf_counter() - t0) * 1000   # P(defective), ms (preprocess + model)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": meta["version"]}


@app.get("/info")
def info():
    return {"model_version": meta["version"], "input_size": meta["size"],
            "threshold": meta["threshold"], "classes": meta["classes"]}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    rid = uuid.uuid4().hex[:8]
    if file.content_type not in ALLOWED:
        log.warning(f"rid={rid} rejected content_type={file.content_type}")
        raise HTTPException(415, "Only JPEG or PNG images are supported")

    data = await file.read(MAX_BYTES + 1)          # never read more than the limit + 1 byte
    if len(data) > MAX_BYTES:
        log.warning(f"rid={rid} rejected: file larger than {MAX_BYTES} bytes")
        raise HTTPException(413, f"File too large (max {MAX_BYTES} bytes)")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        log.warning(f"rid={rid} rejected: undecodable image")
        raise HTTPException(400, "Invalid or corrupt image")
    if max(img.shape[:2]) > MAX_SIDE:
        log.warning(f"rid={rid} rejected: image side > {MAX_SIDE}px")
        raise HTTPException(400, f"Image too large (max {MAX_SIDE}px per side)")

    try:
        p_def, ms = await run_in_threadpool(infer, img)   # keep the event loop free
    except Exception:
        log.exception(f"rid={rid} inference failed")
        raise HTTPException(500, "Inference failed")

    label = "defective" if p_def >= meta["threshold"] else "normal"
    log.info(f"rid={rid} label={label} p_defective={p_def:.4f} latency_ms={ms:.1f}")
    return {"predicted_class": label,
            "confidence": round(p_def if label == "defective" else 1 - p_def, 4),
            "defect_probability": round(p_def, 4),
            "threshold": round(meta["threshold"], 4),
            "model_version": meta["version"],
            "latency_ms": round(ms, 1)}
