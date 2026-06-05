"""
web/backend/main.py
────────────────────────────────────────────────────────────────
Point d'entrée FastAPI — DED-Monitor v4
Toutes les IPs/ports sont lus depuis les variables d'environnement
injectées par START_SYSTEM.bat (source : config.env à la racine).

Variables reconnues (avec valeurs par défaut) :
  BACKEND_HOST  = 172.20.10.4
  BACKEND_PORT  = 8080
  ESP32_IP      = 172.20.10.2
  ESP32_PORT    = 81
  MQTT_HOST     = 172.20.10.4
  MQTT_PORT     = 1883
  MQTT_WS_PORT  = 9091
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Deque

# Suppress TensorFlow verbose startup logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Résolution des chemins ────────────────────────────────────────────────────
_BACKEND_DIR  = Path(__file__).resolve().parent          # web/backend/
_FRONTEND_DIR = _BACKEND_DIR.parent / "frontend"         # web/frontend/

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ── Load root config file if available ───────────────────────────────────────
# This makes config.env work when the web backend is started directly.
def _load_root_config_env() -> None:
    env_path = _BACKEND_DIR.parent.parent / "config.env"
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key and value and os.getenv(key) is None:
                    os.environ[key] = value
        logger = logging.getLogger("startup")
        logger.info("[Main] Loaded config.env from %s", env_path)
    except Exception as exc:
        logger = logging.getLogger("startup")
        logger.warning("[Main] Unable to load config.env: %s", exc)

_load_root_config_env()

# ── Startup diagnostics ───────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt = "%H:%M:%S",
)
_diag_logger = logging.getLogger("startup")
_diag_logger.info("Python executable : %s", sys.executable)
_diag_logger.info("Python version    : %s", sys.version.split()[0])

try:
    import tensorflow as tf
    _diag_logger.info("TensorFlow        : %s ✓", tf.__version__)
except Exception as _tf_err:
    _diag_logger.warning("TensorFlow        : NOT LOADED (%s)", _tf_err)

try:
    import mediapipe as mp
    _diag_logger.info("MediaPipe         : %s (solutions=%s)",
                      getattr(mp, "__version__", "?"),
                      hasattr(mp, "solutions"))
except Exception:
    _diag_logger.info("MediaPipe         : not available (using OpenCV cascade)")

from mqtt_client import MQTTClient
from state import broadcaster, get_snapshot
from vision_processor import (
    VisionProcessor,
    _DB_PATH,
    _db_get_patient_profile,
    _db_list_patient_profiles,
    _db_save_patient_profile,
)
import ml_loader
from predict_router import router as predict_router

# ── Lecture des IPs depuis l'environnement ────────────────────────────────────
_MQTT_HOST   = os.getenv("MQTT_HOST",  "172.20.10.4")
_MQTT_PORT   = int(os.getenv("MQTT_PORT", "1883"))
_ESP32_IP    = os.getenv("ESP32_IP",   "172.20.10.2")
_ESP32_PORT  = os.getenv("ESP32_PORT", "81")
_CAM_STREAM  = os.getenv("CAM_STREAM", f"http://{_ESP32_IP}:{_ESP32_PORT}/stream")
_CNN_MODEL   = os.getenv("CNN_MODEL",  str(_BACKEND_DIR / "best_model.h5"))
_XGB_MODEL   = os.getenv("XGB_MODEL",  str(ml_loader.MODEL_PATH))

# ── Buffers ───────────────────────────────────────────────────────────────────
history_buffer: Deque[dict] = deque(maxlen=50)
last_history_ts: str        = ""
messages_store: list[dict]  = []

logger = logging.getLogger(__name__)

# ── Services ──────────────────────────────────────────────────────────────────
_mqtt_client = MQTTClient(host=_MQTT_HOST, port=_MQTT_PORT)

_vision_processor = VisionProcessor(
    stream_url = _CAM_STREAM,
    cnn_path   = _CNN_MODEL,
    xgb_path   = _XGB_MODEL,
)

WS_HEARTBEAT_TIMEOUT = 25.0

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("[Main] MQTT  → %s:%s", _MQTT_HOST, _MQTT_PORT)
    logger.info("[Main] ESP32 → %s", _CAM_STREAM)
    logger.info("[Main] CNN   → %s", _CNN_MODEL)
    logger.info("[Main] XGB   → %s", _XGB_MODEL)
    # Load model+scaler once into process memory (ml_loader handles fallbacks).
    try:
        ml_loader.load_assets()
        model, scaler = ml_loader.get_assets()
        _vision_processor._xgb_model = model
        logger.info("[Main] ML assets loaded and injected into VisionProcessor")
    except RuntimeError as exc:
        logger.error("[Main] ❌ ML assets unavailable: %s", exc)
        # Re-raise to stop startup (explicit message already logged by ml_loader)
        raise
    broadcaster.set_loop(asyncio.get_event_loop())
    _mqtt_client.start()
    _vision_processor.start()
    _print_banner()
    yield
    _mqtt_client.stop()
    _vision_processor.stop()

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(title="DED-Monitor API v4", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Enregistre les routes de prédiction IA
app.include_router(predict_router)

# ── Servir le frontend statique ───────────────────────────────────────────────
# NOTE : Le frontend (HTML + dist/ TypeScript) est servi par un serveur HTTP
# statique séparé sur le port 8081 (py -m http.server 8081).
# FastAPI sur le port 8000 gère UNIQUEMENT :
#   - Les routes API REST (/api/*)
#   - Le WebSocket (/ws)
#   - Le flux vidéo MJPEG (/api/video_feed)
#
# Les routes ci-dessous sont conservées comme fallback au cas où FastAPI
# serait accédé directement sur le port 8000 (ex: tests, développement).

@app.get("/", include_in_schema=False)
async def serve_index():
    """Fallback — normalement servi par http.server:8081."""
    index = _FRONTEND_DIR / "ded-monitor.html"
    if index.exists():
        return FileResponse(str(index))
    from fastapi.responses import JSONResponse
    return JSONResponse(
        {"info": "Frontend servi sur le port 8081. Accédez à http://HOST:8081/ded-monitor.html"},
        status_code=200,
    )

@app.get("/Config.js", include_in_schema=False)
async def serve_config_js():
    """
    Génère Config.js dynamiquement — utilisé si FastAPI est accédé directement.
    En production normale, Config.js est le fichier statique servi par port 8081.
    """
    backend_host = os.getenv("BACKEND_HOST", "172.20.10.4")
    backend_port = os.getenv("BACKEND_PORT", "8080")
    mqtt_ws_port = os.getenv("MQTT_WS_PORT", "9091")
    content = (
        "/* Auto-généré par FastAPI — ne pas modifier manuellement */\n"
        "window.CONFIG = {\n"
        f'  BACKEND_HOST: "{backend_host}",\n'
        f'  BACKEND_PORT: "{backend_port}",\n'
        f'  MQTT_WS_HOST: "{backend_host}",\n'
        f'  MQTT_WS_PORT: "{mqtt_ws_port}"\n'
        "};\n"
    )
    from fastapi.responses import Response
    return Response(content=content, media_type="application/javascript")

# Monte dist/ comme fallback si accès direct sur port 8000
_DIST_DIR = _FRONTEND_DIR / "dist"
if _DIST_DIR.exists():
    app.mount("/dist", StaticFiles(directory=str(_DIST_DIR)), name="dist")
    logger.info("[Main] dist/ monté → /dist  (%s)", _DIST_DIR)
else:
    logger.warning(
        "[Main] dist/ introuvable (%s). "
        "Lancez 'npx tsc' dans web/frontend/ pour compiler les modules TypeScript.",
        _DIST_DIR,
    )

_PATIENT_PROFILES_FILE = _BACKEND_DIR / "patient_profiles.json"
_MESSAGES_FILE = _BACKEND_DIR / "messages.json"


def _load_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Flux vidéo ────────────────────────────────────────────────────────────────
@app.get("/api/video_feed")
async def video_feed():
    async def generate():
        while True:
            encoded = None
            with _vision_processor.lock:
                frame = _vision_processor.latest_frame
                if frame is not None:
                    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        encoded = buf.tobytes()
            if encoded is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + encoded + b"\r\n"
                )
            await asyncio.sleep(0.04)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/stream")
async def api_stream():
    """Server-Sent Events endpoint for realtime snapshot updates."""
    async def event_generator():
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        broadcaster.add_subscriber(queue)
        try:
            yield f"data: {json.dumps(_sanitize(get_snapshot()))}\n\n"
            while True:
                payload = await queue.get()
                yield f"data: {json.dumps(_sanitize(payload))}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            broadcaster.remove_subscriber(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ── Helpers REST ──────────────────────────────────────────────────────────────
def _sanitize(snap: dict) -> dict:
    snap.setdefault("sensors",    {})
    snap.setdefault("vision",     {})
    snap.setdefault("prediction", {})
    snap.setdefault("status",     {})
    snap["sensors"].setdefault("temperature", 0.0)
    snap["sensors"].setdefault("humidity",    0.0)
    snap["sensors"].setdefault("lux",         0.0)
    snap["sensors"].setdefault("eye_temp",    0.0)
    snap["vision"].setdefault("eye_state",             "unknown")
    snap["vision"].setdefault("blink_count",           0)
    snap["vision"].setdefault("blink_rate_per_minute", 0.0)
    snap["vision"].setdefault("eye_confidence",        None)
    snap["vision"].setdefault("ear",                   -1.0)
    snap["prediction"].setdefault("label",          "En attente")
    snap["prediction"].setdefault("confidence",     0.0)
    snap["prediction"].setdefault("recommendation", "")
    snap["status"].setdefault("camera_online",  False)
    snap["status"].setdefault("mqtt_connected", False)
    snap["status"].setdefault("last_update",    "")
    return snap

# ── Endpoints REST ────────────────────────────────────────────────────────────
@app.get("/api/state")
async def api_state():
    """Alias de compatibilité → /api/status."""
    return await api_status()

@app.get("/api/status")
async def api_status():
    global last_history_ts
    snap       = _sanitize(get_snapshot())
    current_ts = snap["status"].get("last_update", "")
    sensors    = snap["sensors"]

    # Record history when we have meaningful measurement data:
    #  - MQTT sensor values,
    #  - camera/vision state (blink count / eye state),
    #  - or a real AI prediction.
    has_real_data = any(
        sensors.get(k, 0.0) != 0.0
        for k in ("temperature", "humidity", "lux", "eye_temp")
    )
    has_vision_data = (
        snap["vision"]["blink_count"] > 0
        or snap["vision"]["eye_state"] != "unknown"
    )
    has_ai_prediction = snap["prediction"]["label"] not in ("", "En attente")

    if current_ts and current_ts != last_history_ts and (has_real_data or has_vision_data or has_ai_prediction):
        history_buffer.append({
            "timestamp":   current_ts,
            "temperature": sensors["temperature"],
            "humidity":    sensors["humidity"],
            "lux":         sensors["lux"],
            "eye_temp":    sensors["eye_temp"],
            "blink_rate":  snap["vision"]["blink_rate_per_minute"],
            "blink_count": snap["vision"]["blink_count"],
            "eye_state":   snap["vision"]["eye_state"],
            "prediction":  snap["prediction"]["label"],
            "confidence":  snap["prediction"]["confidence"],
        })
        last_history_ts = current_ts
    return snap

@app.get("/api/history")
async def api_history(limit: int = 10):
    try:
        # Current snapshot used as a last-resort fallback if DB + buffer are empty
        snap = _sanitize(get_snapshot())

        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            rows = con.execute(
                "SELECT timestamp, temperature, humidity, lux, eye_temp, blink_rate, blink_count, eye_state, prediction, confidence, recommendation "
                "FROM measurements "
                "ORDER BY timestamp DESC "
                "LIMIT ?",
                (limit,),
            ).fetchall()
            rows_dicts = [
                {
                    "timestamp": row[0],
                    "temperature": row[1],
                    "humidity": row[2],
                    "lux": row[3],
                    "eye_temp": row[4],
                    "blink_rate": row[5],
                    "blink_count": row[6],
                    "eye_state": row[7],
                    "prediction": row[8],
                    "confidence": row[9],
                    "recommendation": row[10],
                }
                for row in rows
            ]
            if rows_dicts:
                return {"records": rows_dicts}

            # If DB empty, try in-memory buffer
            fallback = list(history_buffer)[-limit:]
            if fallback:
                return {"records": fallback}

            # Final fallback: return current snapshot only when it contains meaningful data.
            eye_state = snap.get("vision", {}).get("eye_state", "unknown")
            prediction_label = snap.get("prediction", {}).get("label", "En attente")
            has_snapshot_data = any(
                snap.get("sensors", {}).get(k, 0.0) != 0.0
                for k in ("temperature", "humidity", "lux", "eye_temp")
            ) or snap.get("vision", {}).get("blink_count", 0) > 0 or eye_state != "unknown" or prediction_label != "En attente"

            if has_snapshot_data:
                record = {
                    "timestamp": snap.get("status", {}).get("last_update", ""),
                    "temperature": snap.get("sensors", {}).get("temperature", 0.0),
                    "humidity": snap.get("sensors", {}).get("humidity", 0.0),
                    "lux": snap.get("sensors", {}).get("lux", 0.0),
                    "eye_temp": snap.get("sensors", {}).get("eye_temp", 0.0),
                    "blink_rate": snap.get("vision", {}).get("blink_rate_per_minute", 0.0),
                    "blink_count": snap.get("vision", {}).get("blink_count", 0),
                    "eye_state": eye_state,
                    "prediction": prediction_label,
                    "confidence": snap.get("prediction", {}).get("confidence", 0.0),
                    "recommendation": snap.get("prediction", {}).get("recommendation", ""),
                }
                return {"records": [record]}

            return {"records": []}
    except Exception as exc:
        logger.warning("[Main] /api/history DB error: %s", exc)
        fallback = list(history_buffer)[-limit:]
        if fallback:
            return {"records": fallback}
        snap = _sanitize(get_snapshot())
        eye_state = snap.get("vision", {}).get("eye_state", "unknown")
        prediction_label = snap.get("prediction", {}).get("label", "En attente")
        has_snapshot_data = any(
            snap.get("sensors", {}).get(k, 0.0) != 0.0
            for k in ("temperature", "humidity", "lux", "eye_temp")
        ) or snap.get("vision", {}).get("blink_count", 0) > 0 or eye_state != "unknown" or prediction_label != "En attente"

        if has_snapshot_data:
            record = {
                "timestamp": snap.get("status", {}).get("last_update", ""),
                "temperature": snap.get("sensors", {}).get("temperature", 0.0),
                "humidity": snap.get("sensors", {}).get("humidity", 0.0),
                "lux": snap.get("sensors", {}).get("lux", 0.0),
                "eye_temp": snap.get("sensors", {}).get("eye_temp", 0.0),
                "blink_rate": snap.get("vision", {}).get("blink_rate_per_minute", 0.0),
                "blink_count": snap.get("vision", {}).get("blink_count", 0),
                "eye_state": eye_state,
                "prediction": prediction_label,
                "confidence": snap.get("prediction", {}).get("confidence", 0.0),
                "recommendation": snap.get("prediction", {}).get("recommendation", ""),
            }
            return {"records": [record]}

        return {"records": []}

@app.get("/api/patients")
async def api_patients():
    return {"patients": _db_list_patient_profiles()}

@app.get("/api/patients/{patient_id}")
async def api_patient_detail(patient_id: str):
    profile = _db_get_patient_profile(patient_id)
    if profile is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Patient introuvable")
    return profile

@app.post("/api/patients")
async def api_save_patient_profile(request: Request):
    payload = await request.json()
    code = str(payload.get("id") or payload.get("code") or "").strip()
    name = str(payload.get("name") or "").strip()
    if not code or not name:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Patient code et nom requis")
    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    _db_save_patient_profile(
        code=code,
        name=name,
        age=str(payload.get("age")) if payload.get("age") is not None else None,
        gender=str(payload.get("gender") or ""),
        phone=str(payload.get("phone") or ""),
        treating_doctor=str(payload.get("treatingDoctor") or ""),
        created_at=now,
        updated_at=now,
    )
    return {"status": "created"}

@app.get("/api/messages")
async def api_messages(patient_id: str):
    return [m for m in messages_store if m.get("patient_id") == patient_id]

@app.post("/api/messages")
async def api_send_message(request: Request):
    payload = await request.json()
    messages_store.append(payload)
    return {"status": "created"}

@app.post("/patients/profile")
async def save_patient_profile(request: Request):
    payload = await request.json()
    code = str(payload.get("code") or payload.get("id") or "").strip()
    if not code:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Patient code requis")

    now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()
    profile = {
        "code": code,
        "name": str(payload.get("name") or "").strip(),
        "age": payload.get("age"),
        "sex": str(payload.get("sex") or payload.get("gender") or ""),
        "phone": str(payload.get("phone") or ""),
        "medecin_traitant": str(payload.get("medecin_traitant") or payload.get("treatingDoctor") or ""),
        "created_at": now,
        "updated_at": now,
    }

    profiles = _load_json(_PATIENT_PROFILES_FILE, {})
    if not isinstance(profiles, dict):
        profiles = {}
    profiles[code] = profile
    _save_json(_PATIENT_PROFILES_FILE, profiles)
    return {"status": "saved", "profile": profile}

@app.get("/patients/lookup")
async def lookup_patient(code: str):
    profiles = _load_json(_PATIENT_PROFILES_FILE, {})
    if not isinstance(profiles, dict):
        profiles = {}
    profile = profiles.get(code.strip())
    if profile is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Patient introuvable")
    return profile

@app.get("/patients/list")
async def list_patients():
    profiles = _load_json(_PATIENT_PROFILES_FILE, {})
    if not isinstance(profiles, dict):
        profiles = {}
    return list(profiles.values())

@app.post("/messages/send")
async def send_clinical_message(request: Request):
    payload = await request.json()
    sender = str(payload.get("from") or "").strip()
    recipient = str(payload.get("to") or "").strip()
    text = str(payload.get("text") or "").strip()
    timestamp = str(payload.get("timestamp") or "").strip()
    if not sender or not recipient or not text or not timestamp:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="from,to,text,timestamp requis")

    message = {
        "from": sender,
        "to": recipient,
        "text": text,
        "timestamp": timestamp,
    }

    messages = _load_json(_MESSAGES_FILE, [])
    if not isinstance(messages, list):
        messages = []
    messages.append(message)
    _save_json(_MESSAGES_FILE, messages)
    return {"status": "saved", "message": message}

@app.get("/messages/{patient_code}")
async def get_messages_for_patient(patient_code: str):
    messages = _load_json(_MESSAGES_FILE, [])
    if not isinstance(messages, list):
        messages = []
    return [m for m in messages if str(m.get("to") or "") == patient_code]

@app.get("/api/health")
async def health():
    snap = get_snapshot()
    return {
        "status": "online",
        "mqtt":   snap.get("status", {}).get("mqtt_connected", False),
        "camera": snap.get("status", {}).get("camera_online",  False),
        "esp32_stream": _CAM_STREAM,
        "mqtt_broker":  f"{_MQTT_HOST}:{_MQTT_PORT}",
    }

@app.get("/api/ai_status")
async def ai_status():
    """Retourne le statut des modèles IA chargés (CNN + XGBoost)."""
    cnn_loaded = _vision_processor._cnn_model is not None
    xgb_loaded = _vision_processor._xgb_model is not None
    snap       = get_snapshot()
    return {
        "cnn": {
            "loaded":     cnn_loaded,
            "model_path": _CNN_MODEL,
            "input_shape": (
                f"{_vision_processor._cnn_h}x{_vision_processor._cnn_w}"
                f"x{'1' if _vision_processor._cnn_gray else '3'}"
            ) if cnn_loaded else None,
        },
        "xgboost": {
            "loaded":     xgb_loaded,
            "model_path": _XGB_MODEL,
        },
        "camera_online":  snap.get("status", {}).get("camera_online", False),
        "mqtt_connected": snap.get("status", {}).get("mqtt_connected", False),
        "last_prediction": snap.get("prediction", {}).get("label", "En attente"),
        "last_confidence": snap.get("prediction", {}).get("confidence", 0.0),
        "open_threshold": _vision_processor.get_threshold(),
    }


@app.post("/api/calibrate")
async def calibrate(request: Request):
    """
    Calibration endpoint — two modes:

    Mode 1 (manual): pass pre-computed means
      Body: {"open_mean": 0.72, "closed_mean": 0.28}

    Mode 2 (auto): pass phase + duration, backend collects scores from live stream
      Body: {"phase": "open",   "duration": 2.0}
      Body: {"phase": "closed", "duration": 2.0}
      Body: {"phase": "compute"}  → computes threshold from collected scores

    Threshold = (open_mean + closed_mean) / 2
    """
    payload = await request.json()
    phase = payload.get("phase")

    # ── Mode 1: manual means ──────────────────────────────────────────────
    if "open_mean" in payload and "closed_mean" in payload:
        open_mean   = float(payload["open_mean"])
        closed_mean = float(payload["closed_mean"])
        threshold   = round((open_mean + closed_mean) / 2.0, 3)
        _vision_processor.set_threshold(threshold)
        logger.info("[Calibrate] manual: open=%.3f closed=%.3f → threshold=%.3f",
                    open_mean, closed_mean, threshold)
        return {"status": "ok", "threshold": threshold,
                "open_mean": open_mean, "closed_mean": closed_mean}

    # ── Mode 2: auto collect ──────────────────────────────────────────────
    if phase == "open":
        duration = float(payload.get("duration", 2.0))
        logger.info("[Calibrate] collecting OPEN scores for %.1fs…", duration)
        scores = await asyncio.get_event_loop().run_in_executor(
            None, _vision_processor.collect_scores, duration
        )
        if not scores:
            return {"status": "error", "message": "No face detected during open phase"}
        mean = round(float(sum(scores) / len(scores)), 3)
        logger.info("[Calibrate] open mean=%.3f  (%d frames)", mean, len(scores))
        return {"status": "ok", "phase": "open", "mean": mean, "frames": len(scores)}

    if phase == "closed":
        duration = float(payload.get("duration", 2.0))
        logger.info("[Calibrate] collecting CLOSED scores for %.1fs…", duration)
        scores = await asyncio.get_event_loop().run_in_executor(
            None, _vision_processor.collect_scores, duration
        )
        if not scores:
            return {"status": "error", "message": "No face detected during closed phase"}
        mean = round(float(sum(scores) / len(scores)), 3)
        logger.info("[Calibrate] closed mean=%.3f  (%d frames)", mean, len(scores))
        return {"status": "ok", "phase": "closed", "mean": mean, "frames": len(scores)}

    if phase == "compute":
        open_mean   = float(payload.get("open_mean",   0.70))
        closed_mean = float(payload.get("closed_mean", 0.30))
        threshold   = round((open_mean + closed_mean) / 2.0, 3)
        _vision_processor.set_threshold(threshold)
        logger.info("[Calibrate] compute: open=%.3f closed=%.3f → threshold=%.3f",
                    open_mean, closed_mean, threshold)
        return {"status": "ok", "threshold": threshold,
                "open_mean": open_mean, "closed_mean": closed_mean}

    return {"status": "error", "message": "Invalid payload. Use phase=open/closed/compute or open_mean+closed_mean"}

# ── WebSocket ─────────────────────────────────────────────────────────────────
# CORRECTIF : broadcaster.add_subscriber() déplacé DANS le bloc try
# pour que toute exception soit catchée et loguée (évite le code 1006 silencieux).
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    try:
        # Abonnement au broadcaster DANS le try pour catcher les erreurs
        broadcaster.add_subscriber(queue)

        # Envoi du snapshot initial dès la connexion
        initial = _sanitize(get_snapshot())
        await websocket.send_json(initial)
        logger.info("[WS] ✅ Client connecté — snapshot initial envoyé")

        while True:
            try:
                # Attente événement broadcaster (event-driven)
                msg = await asyncio.wait_for(queue.get(), timeout=WS_HEARTBEAT_TIMEOUT)
                await websocket.send_json(_sanitize(msg))
            except asyncio.TimeoutError:
                # Heartbeat pour maintenir la connexion vivante
                await websocket.send_json({"type": "ping"})

    except WebSocketDisconnect:
        logger.info("[WS] Client déconnecté normalement")
    except Exception as exc:
        # Log complet avec traceback — indispensable pour diagnostiquer le code 1006
        logger.error("[WS] ❌ Erreur inattendue : %s", exc, exc_info=True)
    finally:
        broadcaster.remove_subscriber(queue)

def _print_banner():
    port = os.getenv("BACKEND_PORT", "8080")
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    logger.info(
        "\n%s\n   DED-MONITOR BACKEND v4\n"
        "   API  → http://%s:%s\n"
        "   ESP32→ %s\n"
        "   MQTT → %s:%s\n%s",
        "=" * 50, host, port, _CAM_STREAM,
        _MQTT_HOST, _MQTT_PORT, "=" * 50,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("BACKEND_PORT", "8080")),
        reload=False,
    )