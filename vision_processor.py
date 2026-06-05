"""
vision_processor.py  — v5
══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
──────────────────────────────────────────────────────
  Thread Capture  ──queue──►  Thread CNN
                                ├─ OpenCV Eye Cascade → eye bounding box ratio
                                ├─ EAR proxy (height/width of eye bbox)
                                ├─ CNN (best_model.h5) if TensorFlow available
                                ├─ _tick_blink()    (open→closed transition)
                                ├─ update_vision()  (WebSocket → frontend)
                                └─ _run_xgboost()   (every 5 s)

EAR PROXY (no MediaPipe needed)
──────────────────────────────────────────────────────
  When eye is OPEN  → cascade detects a tall bounding box → ratio > threshold
  When eye is CLOSED → cascade detects nothing OR a flat box → ratio < threshold
  Missing detection for N consecutive frames → eye closed

BLINK COUNTING
──────────────────────────────────────────────────────
  _tick_blink() detects the OPEN→CLOSED transition and increments counter.
  _compute_brpm_unsafe() uses a 60-second sliding window.
"""

import logging
import pickle
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from state import get_snapshot, update_prediction, update_status, update_vision

logger = logging.getLogger(__name__)

import sqlite3
_DB_PATH = "eye_data.db"

try:
    import tensorflow as tf  # noqa: F401  — imported for type resolution
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

def _db_init():
    """Crée les tables de blinks et de mesures si elles n'existent pas — journal WAL pour la concurrence."""
    with sqlite3.connect(_DB_PATH) as con:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("""
            CREATE TABLE IF NOT EXISTS blink_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL    NOT NULL,
                ts_iso    TEXT    NOT NULL,
                blink_no  INTEGER NOT NULL,
                brpm      REAL    NOT NULL
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp      TEXT    NOT NULL,
                temperature    REAL,
                humidity       REAL,
                lux            REAL,
                eye_temp       REAL,
                blink_rate     REAL,
                blink_count    INTEGER,
                eye_state      TEXT,
                prediction     TEXT,
                confidence     REAL,
                recommendation TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS patient_profiles (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                code             TEXT    NOT NULL UNIQUE,
                name             TEXT    NOT NULL,
                age              TEXT,
                gender           TEXT,
                phone            TEXT,
                treating_doctor  TEXT,
                created_at       TEXT    NOT NULL,
                updated_at       TEXT    NOT NULL
            )
        """)
        con.commit()

def _db_log_blink(blink_no: int, brpm: float) -> None:
    """Insère un événement blink (non-bloquant, erreur silencieuse)."""
    try:
        from datetime import datetime, timezone
        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                "INSERT INTO blink_events (ts, ts_iso, blink_no, brpm) VALUES (?,?,?,?)",
                (time.monotonic(), datetime.now(timezone.utc).isoformat(), blink_no, brpm),
            )
            con.commit()
    except Exception as e:
        logger.debug("[Vision] DB log error: %s", e)

def _db_log_measurement(
    timestamp: str,
    temperature: float | None,
    humidity: float | None,
    lux: float | None,
    eye_temp: float | None,
    blink_rate: float,
    blink_count: int,
    eye_state: str,
    prediction: str,
    confidence: float,
    recommendation: str,
) -> None:
    """Insère une mesure persistante dans eye_data.db (non-bloquant)."""
    try:
        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                "INSERT INTO measurements (timestamp, temperature, humidity, lux, eye_temp, blink_rate, blink_count, eye_state, prediction, confidence, recommendation) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    timestamp,
                    temperature,
                    humidity,
                    lux,
                    eye_temp,
                    blink_rate,
                    blink_count,
                    eye_state,
                    prediction,
                    confidence,
                    recommendation,
                ),
            )
            con.commit()
    except Exception as e:
        logger.debug("[Vision] DB log error: %s", e)


def _db_save_patient_profile(
    code: str,
    name: str,
    age: str | None,
    gender: str | None,
    phone: str | None,
    treating_doctor: str | None,
    created_at: str,
    updated_at: str,
) -> None:
    """Insère ou met à jour le profil d'un patient."""
    try:
        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute(
                "INSERT INTO patient_profiles (code, name, age, gender, phone, treating_doctor, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(code) DO UPDATE SET "
                "name=excluded.name, age=excluded.age, gender=excluded.gender, "
                "phone=excluded.phone, treating_doctor=excluded.treating_doctor, "
                "updated_at=excluded.updated_at",
                (code, name, age, gender, phone, treating_doctor, created_at, updated_at),
            )
            con.commit()
    except Exception as e:
        logger.debug("[Vision] DB patient save error: %s", e)


def _db_get_patient_profile(code: str) -> dict | None:
    try:
        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT code, name, age, gender, phone, treating_doctor, created_at, updated_at "
                "FROM patient_profiles WHERE code = ?",
                (code,),
            ).fetchone()
            if row is None:
                return None
            return {
                "role": "patient",
                "id": row["code"],
                "name": row["name"],
                "age": row["age"],
                "gender": row["gender"],
                "phone": row["phone"],
                "treatingDoctor": row["treating_doctor"],
                "createdAt": row["created_at"],
                "updatedAt": row["updated_at"],
            }
    except Exception as e:
        logger.debug("[Vision] DB patient lookup error: %s", e)
        return None


def _db_list_patient_profiles() -> list[dict]:
    try:
        with sqlite3.connect(_DB_PATH, timeout=2) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT code, name, age, gender, phone, treating_doctor, created_at, updated_at "
                "FROM patient_profiles ORDER BY updated_at DESC"
            ).fetchall()
            return [
                {
                    "role": "patient",
                    "id": row["code"],
                    "name": row["name"],
                    "age": row["age"],
                    "gender": row["gender"],
                    "phone": row["phone"],
                    "treatingDoctor": row["treating_doctor"],
                    "createdAt": row["created_at"],
                    "updatedAt": row["updated_at"],
                }
                for row in rows
            ]
    except Exception as e:
        logger.debug("[Vision] DB patient list error: %s", e)
        return []

# ── Constants ──────────────────────────────────────────────────────────────────
ESP32_STREAM_URL      = "http://172.20.10.2:81/stream"
CNN_MODEL_PATH        = "best_model.h5"
XGB_MODEL_PATH        = "model.pkl"
PREDICTION_INTERVAL_S = 5.0
STREAM_RETRY_DELAY_S  = 5.0

# CNN inference constants — match the working Thonny code exactly
IMG_SIZE       = 80
OPEN_THRESHOLD = 0.50   # default; same as Thonny before calibration
SMOOTH_SIZE    = 3       # 3-frame rolling average — fast blink responsive
BLINK_COOLDOWN = 0.25   # seconds between blinks (same as Thonny)
MIN_BLINK_TIME = 0.05   # minimum blink duration in seconds

# Calibration and stabilization constants
CALIBRATION_FRAMES   = 40
CLOSED_RATIO         = 0.82
DEADBAND_RATIO       = 0.04
CNN_CLOSED_THRESH    = 0.42
CNN_OPEN_THRESH      = 0.58
EAR_CLOSED_THRESH    = 0.18
EAR_OPEN_THRESH      = 0.22
EAR_CONSEC_FRAMES    = 3
CLOSED_HOLD_FRAMES   = 2
RECENCY_WEIGHT       = 2.0
INTENSITY_ALPHA      = 0.002

# MediaPipe face-mesh indices for EAR and eye ROI extraction.
_L_EAR_IDX = [33, 160, 158, 133, 153, 144]
_R_EAR_IDX = [362, 385, 387, 263, 373, 380]
_L_EYE_HULL = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
_R_EYE_HULL = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 375, 380, 381, 382]

# CLAHE — created once, reused every frame (same as Thonny)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

# ── OpenCV Face Cascade ────────────────────────────────────────────────────────
# Use haarcascade_frontalface_default.xml — same as the working Thonny code.
_CASCADE_FACE: Optional[cv2.CascadeClassifier] = None
_CASCADE_EYES: Optional[cv2.CascadeClassifier] = None
_USE_MEDIAPIPE = False
_mp_face_mesh  = None

try:
    base = cv2.data.haarcascades
    fc   = cv2.CascadeClassifier(f"{base}haarcascade_frontalface_default.xml")
    ec   = cv2.CascadeClassifier(f"{base}haarcascade_eye.xml")
    _CASCADE_FACE = None if fc.empty() else fc
    _CASCADE_EYES = None if ec.empty() else ec
    logger.info(
        "[Vision] OpenCV cascades — face=%s  eye=%s",
        "OK" if _CASCADE_FACE else "MISSING",
        "OK" if _CASCADE_EYES else "MISSING",
    )
except Exception as e:
    logger.warning("[Vision] Cascade load error: %s", e)


# ── Sensor helper ──────────────────────────────────────────────────────────────
def _sensor_val(value, default: float) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        return None if v == 0.0 else v
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  EAR COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ear(landmarks, indices: list, w: int, h: int) -> float:
    """Eye Aspect Ratio from 6 MediaPipe landmarks."""
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float32,
    )
    # Vertical distances
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
#  EYE ROI EXTRACTION (for CNN)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_eye_roi_mediapipe(
    frame: np.ndarray,
    landmarks,
    hull_indices: list,
    h_model: int,
    w_model: int,
    grayscale: bool,
    padding: float = 0.30,
) -> Optional[np.ndarray]:
    """
    Extract eye ROI using MediaPipe landmark bounding box.
    Returns tensor shaped (1, H, W, 1) or (1, H, W, 3), or None on error.
    """
    try:
        fh, fw = frame.shape[:2]
        pts = np.array(
            [(int(landmarks[i].x * fw), int(landmarks[i].y * fh))
             for i in hull_indices],
            dtype=np.int32,
        )
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        # Add padding
        pw = int((x2 - x1) * padding)
        ph = int((y2 - y1) * padding)
        x1 = max(0, x1 - pw)
        y1 = max(0, y1 - ph)
        x2 = min(fw, x2 + pw)
        y2 = min(fh, y2 + ph)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        resized = cv2.resize(gray, (w_model, h_model))
        arr = resized.astype("float32") / 255.0

        if grayscale:
            return arr[np.newaxis, :, :, np.newaxis]       # (1,H,W,1)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR).astype("float32") / 255.0
            return rgb[np.newaxis]                          # (1,H,W,3)

    except Exception as e:
        logger.debug(f"[Vision] _extract_eye_roi_mediapipe: {e}")
        return None


def _extract_eye_roi_haar(
    frame: np.ndarray,
    h_model: int,
    w_model: int,
    grayscale: bool,
) -> Tuple[Optional[np.ndarray], str]:
    """Haar-based ROI extraction (fallback when MediaPipe absent)."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        roi_gray = None
        source   = "fixed"

        if _CASCADE_FACE is not None:
            faces = _CASCADE_FACE.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=2, minSize=(30, 30)
            )
            if len(faces) > 0:
                fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                face_gray = gray[fy:fy+fh, fx:fx+fw]

                if _CASCADE_EYES is not None:
                    upper = face_gray[:int(fh * 0.60), :]
                    eyes  = _CASCADE_EYES.detectMultiScale(
                        upper, scaleFactor=1.05, minNeighbors=2, minSize=(8, 8)
                    )
                    if len(eyes) > 0:
                        ex, ey, ew, eh = max(eyes, key=lambda r: r[2] * r[3])
                        roi_gray = upper[ey:ey+eh, ex:ex+ew]
                        source   = "haar_eye"

                if roi_gray is None or roi_gray.size == 0:
                    roi_gray = face_gray[int(fh*0.15):int(fh*0.55), :]
                    source   = "haar_face"

        if roi_gray is None or roi_gray.size == 0:
            fh, fw = gray.shape
            roi_gray = gray[int(fh*0.15):int(fh*0.50), int(fw*0.20):int(fw*0.80)]
            source   = "fixed"

        resized = cv2.resize(roi_gray, (w_model, h_model))
        arr = resized.astype("float32") / 255.0
        if grayscale:
            return arr[np.newaxis, :, :, np.newaxis], source
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR).astype("float32") / 255.0
            return rgb[np.newaxis], source

    except Exception as e:
        logger.debug(f"[Vision] _extract_eye_roi_haar: {e}")
        return None, "error"


def _extract_center_roi(
    frame: np.ndarray,
    h_model: int,
    w_model: int,
    grayscale: bool,
    scale: float = 0.35,
) -> Optional[np.ndarray]:
    """Fallback center-frame ROI when face/eye detection fails."""
    try:
        fh, fw = frame.shape[:2]
        size = int(min(fh, fw) * scale)
        if size < 16:
            return None

        cx, cy = fw // 2, fh // 2
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        x2 = min(fw, cx + size // 2)
        y2 = min(fh, cy + size // 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (w_model, h_model))
        arr = resized.astype("float32") / 255.0

        if grayscale:
            return arr[np.newaxis, :, :, np.newaxis]

        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR).astype("float32") / 255.0
        return rgb[np.newaxis]
    except Exception as e:
        logger.debug(f"[Vision] _extract_center_roi: {e}")
        return None

def _classify_center_frame(frame: np.ndarray, previous_state: str = "open") -> Tuple[str, float]:
    """Fallback open/closed estimate using a central intensity + edge ROI."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        roi = gray[h // 5: 3 * h // 5, w // 4: 3 * w // 4]
        if roi.size == 0:
            return previous_state, 0.50

        roi_avg = float(cv2.mean(roi)[0])
        frame_avg = float(cv2.mean(gray)[0])
        edges = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = float(np.mean(np.abs(edges)))

        if roi_avg < frame_avg * 0.65 or edge_strength < 7.0:
            state = "closed"
            conf = min(0.90, 0.55 + max(0.0, (frame_avg * 0.65 - roi_avg) / 80.0) + max(0.0, (7.0 - edge_strength) / 30.0))
        elif roi_avg > frame_avg * 0.85 or edge_strength > 14.0:
            state = "open"
            conf = min(0.90, 0.55 + max(0.0, (roi_avg - frame_avg * 0.85) / 80.0) + max(0.0, (edge_strength - 14.0) / 30.0))
        else:
            delta = roi_avg - frame_avg          # ← définir delta AVANT usage
            state = "open" if delta >= 0 else "closed"
            conf = 0.50

        logger.debug(
            "[Vision] Center fallback roi_avg=%.1f frame_avg=%.1f edge=%.2f delta=%.1f state=%s",
            roi_avg, frame_avg, edge_strength, delta, state,
        )
        return state, round(conf, 3)

    except Exception as e:
        logger.debug(f"[Vision] _classify_center_frame: {e}")
        return previous_state, 0.50

# ══════════════════════════════════════════════════════════════════════════════
#  VISION PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class VisionProcessor:

    def __init__(
        self,
        stream_url: str = ESP32_STREAM_URL,
        cnn_path:   str = CNN_MODEL_PATH,
        xgb_path:   str = XGB_MODEL_PATH,
    ):
        self.stream_url = stream_url
        self.cnn_path   = cnn_path
        self.xgb_path   = xgb_path

        # Shared frame for FastAPI MJPEG proxy
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        # Single queue: capture → CNN (maxsize=1 keeps only latest frame)
        self._cnn_queue: queue.Queue = queue.Queue(maxsize=1)

        # Models
        self._cnn_model = None
        self._xgb_model = None
        self._cnn_h     = 80
        self._cnn_w     = 80
        self._cnn_gray  = True

        # ── Blink state machine ────────────────────────────────────────────
        self._ear_consec_closed: int   = 0
        self._eye_is_closed:     bool  = False
        self._prev_eye_state:    str   = "open"
        self._prev_closed:       bool  = False

        self._blink_count:     int   = 0
        self._blink_times:     deque = deque()
        self._last_blink_time: float = 0.0
        self._last_known_state: str   = "open"
        self._blink_lock             = threading.Lock()

        # ── Thonny-compatible inference state ─────────────────────────────
        # 4-frame rolling average (same as working Thonny code)
        self._pred_buffer:  list  = []
        # Blink timing (same as Thonny)
        self._eye_closed_flag: bool  = False
        self._blink_start:     float = 0.0

        # XGBoost timing (managed by _xgb_loop thread)
        self._cnn_loop_count: int = 0
        self._frame_counter: int = 0

        # Intensity-based blink detector state (used when CNN + MediaPipe absent)
        self._intensity_baseline: float = -1.0
        self._intensity_history: deque  = deque(maxlen=CALIBRATION_FRAMES * 2)
        self._calibrated: bool = False
        self._closed_threshold: float = 0.0
        self._upper_deadband: float = 0.0
        self._lower_deadband: float = 0.0
        self._adaptive_open_ema: float = 0.0
        self._closed_hold_counter: int = 0
        self._state_history: deque = deque(maxlen=3)
        self._stable_eye_state: str = "open"
        self._smoothed_conf: float = 0.5

        # Cascade-based detector state
        self._cascade_no_eye_frames: int = 0

        # Thread control
        self._stop_evt    = threading.Event()
        self._cap_thread: Optional[threading.Thread] = None
        self._cnn_thread: Optional[threading.Thread] = None
        self._xgb_thread: Optional[threading.Thread] = None  # runs XGBoost independently of camera

    def set_threshold(self, threshold: float) -> None:
        """Set the open/closed threshold from calibration. Thread-safe."""
        global OPEN_THRESHOLD
        OPEN_THRESHOLD = round(threshold, 3)
        self._pred_buffer.clear()  # reset buffer after calibration
        logger.info("[Vision] Threshold updated to %.3f", OPEN_THRESHOLD)

    def get_threshold(self) -> float:
        return OPEN_THRESHOLD

    def _reset_intensity_calibration(self) -> None:
        self._intensity_history.clear()
        self._calibrated = False
        self._closed_threshold = 0.0
        self._upper_deadband = 0.0
        self._lower_deadband = 0.0
        self._adaptive_open_ema = 0.0
        self._closed_hold_counter = 0
        self._stable_eye_state = "open"
        self._smoothed_conf = 0.5

    def _collect_intensity_calibration(self, score: float) -> None:
        if score <= 0.0:
            return
        self._intensity_history.append(score)
        if len(self._intensity_history) < CALIBRATION_FRAMES:
            return

        samples = sorted(self._intensity_history)
        trim = max(1, len(samples) // 10)
        trimmed = samples[trim:-trim] if len(samples) > 2 * trim else samples

        self._intensity_baseline = float(np.mean(trimmed))
        self._closed_threshold = self._intensity_baseline * CLOSED_RATIO
        deadband = self._intensity_baseline * DEADBAND_RATIO
        self._upper_deadband = self._closed_threshold + deadband
        self._lower_deadband = self._closed_threshold - deadband
        self._adaptive_open_ema = self._intensity_baseline
        self._calibrated = True

        logger.info(
            "[Vision] Intensity calibration finished: baseline=%.3f closed_thresh=%.3f deadband=[%.3f, %.3f]",
            self._intensity_baseline,
            self._closed_threshold,
            self._lower_deadband,
            self._upper_deadband,
        )

    def _brightness_from_calibration(self, score: float) -> Tuple[str, float]:
        if score > self._upper_deadband:
            span = max(1e-6, self._intensity_baseline - self._upper_deadband)
            conf = 0.60 + 0.35 * min(1.0, (score - self._upper_deadband) / span)
            return "open", min(0.95, conf)

        if score < self._lower_deadband:
            span = max(1e-6, self._lower_deadband)
            conf = 0.60 + 0.35 * min(1.0, (self._lower_deadband - score) / span)
            return "closed", min(0.95, conf)

        held_state = self._stable_eye_state or self._prev_eye_state or "open"
        return held_state, 0.45

    def _update_adaptive_threshold(self, score: float, current_state: str) -> None:
        if not self._calibrated:
            return
        if current_state == "open" and score > self._upper_deadband * 1.05:
            self._adaptive_open_ema = (
                (1.0 - INTENSITY_ALPHA) * self._adaptive_open_ema
                + INTENSITY_ALPHA * score
            )
            new_baseline = self._adaptive_open_ema
            self._closed_threshold = new_baseline * CLOSED_RATIO
            deadband = new_baseline * DEADBAND_RATIO
            self._upper_deadband = self._closed_threshold + deadband
            self._lower_deadband = self._closed_threshold - deadband

    def _intensity_score(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        y1 = int(h * 0.20)
        y2 = int(h * 0.50)
        x1 = int(w * 0.25)
        x2 = int(w * 0.75)
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0
        return float(cv2.mean(roi)[0]) / 255.0

    def _stabilize_eye_state(self, state: str, conf: float) -> Tuple[str, float]:
        if state == "unknown":
            return self._stable_eye_state or "unknown", 0.0

        self._state_history.append((state, conf))
        votes = {"open": 0.0, "closed": 0.0}
        n = len(self._state_history)
        for i, (s, c) in enumerate(self._state_history):
            if s not in votes:
                continue
            weight = RECENCY_WEIGHT if i == n - 1 else 1.0
            votes[s] += weight * c

        winner = "open" if votes["open"] >= votes["closed"] else "closed"

        if self._closed_hold_counter > 0:
            self._closed_hold_counter -= 1
            winner = "closed"
            logger.debug("[Vision] Closed-hold active (%d remaining)", self._closed_hold_counter)
        elif winner == "closed" and self._stable_eye_state != "closed":
            self._closed_hold_counter = CLOSED_HOLD_FRAMES
            logger.debug("[Vision] Entering closed-hold for %d frames", CLOSED_HOLD_FRAMES)

        if winner != self._stable_eye_state:
            logger.debug("[Vision] Eye state: %s → %s", self._stable_eye_state, winner)
        self._stable_eye_state = winner

        raw_conf = votes[winner] / max(1.0, sum(votes.values()))
        self._smoothed_conf = 0.7 * self._smoothed_conf + 0.3 * raw_conf
        return winner, round(self._smoothed_conf, 3)

    def collect_scores(self, duration_s: float = 2.0) -> list:
        """
        Collect raw CNN scores for `duration_s` seconds from the live stream.
        Used by the calibration endpoint to compute open/closed means.
        Returns a list of float scores (empty if no camera).
        """
        scores: list = []
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.05)
                continue
            if self._cnn_model is None:
                time.sleep(0.05)
                continue
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                fh, fw = gray.shape[:2]
                if _CASCADE_FACE is not None:
                    gray_eq = cv2.equalizeHist(gray)
                    faces = _CASCADE_FACE.detectMultiScale(
                        gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                    )
                    if len(faces) > 0:
                        fx, fy, fw_f, fh_f = max(faces, key=lambda r: r[2] * r[3])
                        fx = max(0, fx); fy = max(0, fy)
                        fw_f = min(fw_f, fw - fx); fh_f = min(fh_f, fh - fy)
                        face_gray = gray[fy:fy + fh_f, fx:fx + fw_f]
                        efh, efw = face_gray.shape[:2]
                        y1 = int(efh * 0.18); y2 = int(efh * 0.52)
                        lx1 = int(efw * 0.04); lx2 = int(efw * 0.46)
                        rx1 = int(efw * 0.54); rx2 = int(efw * 0.96)
                        eyes = []
                        for eye_gray in (face_gray[y1:y2, lx1:lx2],
                                         face_gray[y1:y2, rx1:rx2]):
                            if eye_gray.size == 0 or eye_gray.shape[0] < 5:
                                continue
                            resized = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
                            clahe   = _CLAHE.apply(resized)
                            arr     = clahe.astype(np.float32) / 255.0
                            eyes.append(arr.reshape(IMG_SIZE, IMG_SIZE, 1))
                        if eyes:
                            batch = np.stack(eyes, axis=0)
                            preds = self._cnn_model(batch, training=False).numpy()
                            scores.append(float(np.mean(preds)))
            except Exception:
                pass
            time.sleep(0.05)
        return scores

    def start(self) -> None:
        _db_init()
        self._stop_evt.clear()
        self._load_models()
        self._reset_intensity_calibration()
        self._cap_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="VisionCapture"
        )
        self._cnn_thread = threading.Thread(
            target=self._cnn_loop, daemon=True, name="VisionCNN"
        )
        self._xgb_thread = threading.Thread(
            target=self._xgb_loop, daemon=True, name="VisionXGB"
        )
        self._cap_thread.start()
        self._cnn_thread.start()
        self._xgb_thread.start()
        logger.info(
            "[Vision] Started — MediaPipe=%s  CNN=%s  XGB=%s",
            _USE_MEDIAPIPE,
            self._cnn_model is not None,
            self._xgb_model is not None,
        )

    def stop(self) -> None:
        self._stop_evt.set()
        for t in (self._cap_thread, self._cnn_thread, self._xgb_thread):
            if t:
                t.join(timeout=3)
        if _mp_face_mesh is not None:
            try:
                _mp_face_mesh.close()
            except Exception:
                pass

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        # ── CNN ───────────────────────────────────────────────────────────
        try:
            import tensorflow as tf
            tf_load = tf.keras.models.load_model
            p = Path(self.cnn_path)
            if p.exists():
                try:
                    self._cnn_model = tf_load(str(p), compile=False)
                    _, h, w, c      = self._cnn_model.input_shape
                    self._cnn_h     = int(h) if h else 80
                    self._cnn_w     = int(w) if w else 80
                    self._cnn_gray  = (int(c) == 1)
                    logger.info(
                        "[Vision] CNN chargé — input %dx%d %s",
                        self._cnn_h, self._cnn_w,
                        "gray" if self._cnn_gray else "rgb",
                    )
                except Exception as cnn_err:
                    # Couvre explicitement "file signature not found" (H5 corrompu)
                    # et tout autre échec de désérialisation Keras/TF.
                    logger.error(
                        "[Vision] Échec chargement CNN (%s) — "
                        "bascule en mode heuristique. Vérifiez l'intégrité de '%s'.",
                        cnn_err, self.cnn_path,
                    )
                    self._cnn_model = None
                    # Informe le frontend que la caméra n'est pas opérationnelle au sens ML
                    update_status(camera_online=False)
            else:
                logger.warning("[Vision] Fichier CNN introuvable : %s", self.cnn_path)
        except ImportError:
            logger.warning("[Vision] TensorFlow absent — CNN désactivé.")
        except Exception as e:
            logger.error("[Vision] Erreur inattendue au chargement CNN : %s", e)

        # ── XGBoost ───────────────────────────────────────────────────────
        try:
            # If an XGBoost instance was injected (e.g. by main via ml_loader),
            # reuse it instead of reloading from disk to avoid duplicate
            # deserialisation and keep the model resident in RAM.
            if self._xgb_model is not None:
                logger.info("[Vision] XGBoost instance already present in-memory; skipping disk load.")
            else:
                p = Path(self.xgb_path)
                if p.exists():
                    with open(p, "rb") as f:
                        self._xgb_model = pickle.load(f)
                    logger.info("[Vision] XGBoost chargé : %s", self.xgb_path)
                else:
                    logger.warning("[Vision] Fichier XGBoost introuvable : %s", self.xgb_path)
        except Exception as e:
            logger.warning("[Vision] Erreur chargement XGBoost : %s", e)
    # ── Thread 1: Capture ──────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Connect to ESP32 stream, push frames to CNN queue."""
        while not self._stop_evt.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    raise IOError(f"Cannot open stream: {self.stream_url}")

                update_status(camera_online=True)
                logger.info("[Vision] Stream open: %s", self.stream_url)

                while not self._stop_evt.is_set():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning("[Vision] Empty frame — reconnecting…")
                        break

                    with self.lock:
                        self.latest_frame = frame.copy()

                    # Drop if CNN thread is busy (always keep newest frame)
                    try:
                        self._cnn_queue.put_nowait(frame)
                    except queue.Full:
                        pass

            except Exception as e:
                logger.error("[Vision] Stream error: %s", e)
                update_status(camera_online=False)
            finally:
                if cap:
                    cap.release()

            if not self._stop_evt.is_set():
                update_status(camera_online=False)
                logger.info("[Vision] Reconnecting in %ss…", STREAM_RETRY_DELAY_S)
                time.sleep(STREAM_RETRY_DELAY_S)

    # ── Thread 2: CNN inference (single source of truth) ──────────────────────

    def _cnn_loop(self) -> None:
        """
        For each frame:
          1. Run MediaPipe → get face landmarks
          2. Compute EAR → primary eye state
          3. Extract eye ROI → run CNN → get confidence
          4. Tick blink counter on EAR transition
          5. Push update to WebSocket
          6. Every 5s: run XGBoost
        """
        """Boucle principale de traitement vision."""
        while not self._stop_evt.is_set():
            try:
                frame = self._cnn_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._cnn_loop_count += 1

            # 1. CNN inference — same pipeline as working Thonny code
            eye_state, eye_conf, ear_value = self._process_frame(frame)

            # 2. Blink detection — closed→open transition counting
            #    This follows the original Thonny-style blink detector and avoids
            #    spurious counts when the eye state flickers during threshold tuning.
            current_time = time.monotonic()

            if eye_state in ("open", "closed"):
                if eye_state == "closed":
                    if not self._eye_closed_flag:
                        self._eye_closed_flag = True
                        self._blink_start = current_time
                else:  # eye_state == "open"
                    if self._eye_closed_flag:
                        duration = current_time - self._blink_start
                        if (duration > MIN_BLINK_TIME and
                                (current_time - self._last_blink_time) > BLINK_COOLDOWN):
                            self._blink_count += 1
                            self._last_blink_time = current_time
                            now_ts = current_time
                            self._blink_times.append(now_ts)
                            cutoff = now_ts - 60.0
                            while self._blink_times and self._blink_times[0] < cutoff:
                                self._blink_times.popleft()
                            brpm_log = self._compute_brpm_unsafe()
                            _db_log_blink(self._blink_count, brpm_log)
                            logger.info("[Vision] BLINK ✓  Total=%d  BRPM=%.1f",
                                        self._blink_count, brpm_log)
                        self._eye_closed_flag = False
            else:
                # Face lost — reset blink state and buffer
                self._eye_closed_flag = False
                self._pred_buffer.clear()

            # 3. Blink rate
            brpm = self._compute_brpm_unsafe()

            # 4. Update state for frontend
            update_vision(
                eye_state      = eye_state,
                eye_confidence = round(eye_conf, 3) if eye_conf > 0 else None,
                blink_count    = self._blink_count,
                blink_rate     = brpm,
                ear            = ear_value,
            )

            # 5. Periodic log
            if self._cnn_loop_count % 10 == 0:
                logger.info(
                    "[Vision] Loop #%d | State: %s | Blinks: %d | BRPM: %.1f",
                    self._cnn_loop_count, eye_state, self._blink_count, brpm
                )

    # ── Core processing ────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> Tuple[str, float, float]:
        """
        Returns (eye_state, confidence, ear).

        eye_state  : "open" | "closed" | "no_face"
        confidence : 0.0–1.0  (from CNN, or derived from EAR distance)
        ear        : raw EAR value, -1.0 if no face detected
        """
        # ── MediaPipe path (preferred) ─────────────────────────────────────
        if _USE_MEDIAPIPE and _mp_face_mesh is not None:
            return self._process_mediapipe(frame)

        # ── Haar fallback ──────────────────────────────────────────────────
        return self._process_haar(frame)

    # ── MediaPipe processing ───────────────────────────────────────────────────

    def _process_mediapipe(self, frame: np.ndarray) -> Tuple[str, float, float]:
        fh, fw = frame.shape[:2]

        # MediaPipe needs RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _mp_face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            logger.debug("[Vision] MediaPipe landmarks missing — trying Haar ROI fallback")
            tensor, source = _extract_eye_roi_haar(
                frame, self._cnn_h, self._cnn_w, self._cnn_gray
            )
            if tensor is None:
                logger.debug("[Vision] Haar ROI fallback failed — cropping center ROI")
                tensor = _extract_center_roi(frame, self._cnn_h, self._cnn_w, self._cnn_gray)
                source = "center"

            if tensor is not None and self._cnn_model is not None:
                try:
                    preds     = self._cnn_model(tensor, training=False).numpy()
                    prob      = float(preds[0, 0])
                    if prob > CNN_OPEN_THRESH:
                        raw_state = "open"
                        raw_conf  = prob
                    elif prob < CNN_CLOSED_THRESH:
                        raw_state = "closed"
                        raw_conf  = 1.0 - prob
                    else:
                        score = self._intensity_score(frame)
                        raw_state, raw_conf = self._brightness_from_calibration(score)

                    logger.debug(
                        "[Vision] Fallback ROI source=%s prob=%.3f raw_state=%s raw_conf=%.3f",
                        source, prob, raw_state, raw_conf,
                    )
                    stabilized_state, stabilized_conf = self._stabilize_eye_state(raw_state, raw_conf)
                    return stabilized_state, round(stabilized_conf, 3), -1.0
                except Exception as e:
                    logger.warning("[Vision] CNN fallback predict error: %s", e)

            state, conf = _classify_center_frame(frame, self._prev_eye_state)
            logger.debug("[Vision] Center-frame fallback classification=%s conf=%.3f", state, conf)
            stabilized_state, stabilized_conf = self._stabilize_eye_state(state, conf)
            return stabilized_state, round(stabilized_conf, 3), -1.0

        lm = results.multi_face_landmarks[0].landmark

        # ── EAR calculation ────────────────────────────────────────────────
        left_ear  = _ear(lm, _L_EAR_IDX, fw, fh)
        right_ear = _ear(lm, _R_EAR_IDX, fw, fh)
        avg_ear   = (left_ear + right_ear) / 2.0

        # EAR-based state  with hysteresis around the open/closed gap.
        if avg_ear < EAR_CLOSED_THRESH:
            self._ear_consec_closed += 1
        elif avg_ear > EAR_OPEN_THRESH:
            self._ear_consec_closed = 0

        if self._ear_consec_closed >= EAR_CONSEC_FRAMES:
            ear_state = "closed"
        elif avg_ear > EAR_OPEN_THRESH:
            ear_state = "open"
        else:
            ear_state = "closed" if self._eye_is_closed else "open"

        self._eye_is_closed = (ear_state == "closed")
        self._prev_eye_state = ear_state

        # EAR confidence: how far from the midpoint threshold
        mid_thresh = (EAR_OPEN_THRESH + EAR_CLOSED_THRESH) / 2.0
        ear_conf   = min(1.0, abs(avg_ear - mid_thresh) / mid_thresh)
        logger.debug(
            "[Vision] EAR left=%.3f right=%.3f avg=%.3f ear_state=%s closed_count=%d",
            left_ear, right_ear, avg_ear, ear_state, self._ear_consec_closed,
        )

        # ── CNN on extracted eye ROI ───────────────────────────────────────
        cnn_state = None
        cnn_conf  = 0.0

        if self._cnn_model is not None:
            # Use the left eye ROI (or average both — here we use left)
            tensor = _extract_eye_roi_mediapipe(
                frame, lm, _L_EYE_HULL,
                self._cnn_h, self._cnn_w, self._cnn_gray,
            )
            if tensor is None:
                logger.debug("[Vision] MediaPipe ROI failed — attempting center-frame fallback")
                tensor = _extract_center_roi(frame, self._cnn_h, self._cnn_w, self._cnn_gray)

            if tensor is not None:
                try:
                    preds     = self._cnn_model(tensor, training=False).numpy()
                    prob      = float(preds[0, 0])
                    if prob > CNN_OPEN_THRESH:
                        cnn_state = "open"
                        cnn_conf  = prob
                    elif prob < CNN_CLOSED_THRESH:
                        cnn_state = "closed"
                        cnn_conf  = 1.0 - prob
                    else:
                        score = self._intensity_score(frame)
                        cnn_state, cnn_conf = self._brightness_from_calibration(score)
                except Exception as e:
                    logger.warning("[Vision] CNN predict error: %s", e)

        # ── Combine EAR + CNN ──────────────────────────────────────────────
        raw_state = ear_state
        raw_conf  = ear_conf

        if cnn_state is not None:
            if cnn_state == ear_state:
                raw_conf = min(1.0, (ear_conf + cnn_conf) / 2.0 + 0.15)
            else:
                raw_conf = max(0.05, ear_conf * 0.60)

        stabilized_state, stabilized_conf = self._stabilize_eye_state(raw_state, raw_conf)
        return stabilized_state, round(stabilized_conf, 3), round(avg_ear, 4)

    # ── Primary vision processing — matches working Thonny code exactly ─────────

    def _process_haar(self, frame: np.ndarray) -> Tuple[str, float, float]:
        """
        Eye state detection using the exact same pipeline as the working Thonny code:
          1. Detect face with haarcascade_frontalface_default.xml
          2. Crop eye region using fixed ratios from face bbox (same as Thonny)
          3. Apply CLAHE + resize to 80×80 (same as Thonny)
          4. Run CNN on both eye crops, average the scores
          5. Apply 4-frame rolling average (same as Thonny)
          6. Threshold: avg >= OPEN_THRESHOLD → open, else → closed

        Returns (eye_state, confidence, ear=-1.0)
        """
        self._frame_counter += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        # ── Face detection ─────────────────────────────────────────────────
        left_eye_gray  = None
        right_eye_gray = None
        face_found     = False

        if _CASCADE_FACE is not None:
            gray_eq = cv2.equalizeHist(gray)
            faces = _CASCADE_FACE.detectMultiScale(
                gray_eq,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
            )
            if len(faces) > 0:
                fx, fy, fw_f, fh_f = max(faces, key=lambda r: r[2] * r[3])
                # Clamp to frame bounds
                fx = max(0, fx); fy = max(0, fy)
                fw_f = min(fw_f, fw - fx); fh_f = min(fh_f, fh - fy)
                face_gray = gray[fy:fy + fh_f, fx:fx + fw_f]

                # Eye crop using same fixed ratios as Thonny code
                efh, efw = face_gray.shape[:2]
                y1 = int(efh * 0.18); y2 = int(efh * 0.52)
                lx1 = int(efw * 0.04); lx2 = int(efw * 0.46)
                rx1 = int(efw * 0.54); rx2 = int(efw * 0.96)

                left_eye_gray  = face_gray[y1:y2, lx1:lx2]
                right_eye_gray = face_gray[y1:y2, rx1:rx2]
                face_found = True

        # ── CNN inference ──────────────────────────────────────────────────
        if self._cnn_model is not None and face_found:
            try:
                eyes = []
                for eye_gray in (left_eye_gray, right_eye_gray):
                    if eye_gray is None or eye_gray.size == 0:
                        continue
                    if eye_gray.shape[0] < 5 or eye_gray.shape[1] < 5:
                        continue
                    # CLAHE + resize — exact same as Thonny preprocess()
                    resized = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
                    clahe   = _CLAHE.apply(resized)
                    arr     = clahe.astype(np.float32) / 255.0
                    eyes.append(arr.reshape(IMG_SIZE, IMG_SIZE, 1))

                if eyes:
                    batch = np.stack(eyes, axis=0)   # (N, 80, 80, 1)
                    preds = self._cnn_model(batch, training=False).numpy()
                    raw   = float(np.mean(preds))

                    # 4-frame rolling average — same as Thonny
                    self._pred_buffer.append(raw)
                    if len(self._pred_buffer) > SMOOTH_SIZE:
                        self._pred_buffer.pop(0)
                    avg_pred = float(np.mean(self._pred_buffer))

                    if avg_pred > CNN_OPEN_THRESH:
                        raw_state = "open"
                        raw_conf = avg_pred
                    elif avg_pred < CNN_CLOSED_THRESH:
                        raw_state = "closed"
                        raw_conf = 1.0 - avg_pred
                    else:
                        score = self._intensity_score(frame)
                        raw_state, raw_conf = self._brightness_from_calibration(score)

                    self._update_adaptive_threshold(self._intensity_score(frame), raw_state)
                    state, conf = self._stabilize_eye_state(raw_state, raw_conf)

                    if self._frame_counter % 15 == 0:
                        logger.info(
                            "[Vision] Frame %4d | face=YES | raw=%.3f avg=%.3f → %-6s",
                            self._frame_counter, raw, avg_pred, state,
                        )

                    self._prev_eye_state = state
                    return state, round(conf, 3), -1.0

            except Exception as e:
                logger.warning("[Vision] CNN predict error: %s", e)

        # ── No face detected ───────────────────────────────────────────────
        if not face_found:
            if self._frame_counter % 30 == 0:
                logger.info("[Vision] Frame %4d | no face detected", self._frame_counter)
            # Clear buffer when face lost — same as Thonny
            self._pred_buffer.clear()
            self._prev_eye_state = "unknown"
            return "unknown", 0.0, -1.0

        # ── CNN not loaded — intensity fallback ────────────────────────────
        state, conf = self._intensity_blink_detect(frame)
        self._prev_eye_state = state
        return state, round(conf, 3), -1.0
    # ── Intensity-based blink detector (no CNN, no MediaPipe) ─────────────────

    def _intensity_blink_detect(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Detects eye open/closed state purely from pixel intensity in the
        upper-center region of the frame (where eyes sit on a face).

        Algorithm:
          1. Crop the eye-region band (rows 20%–50%, cols 25%–75%)
          2. Compute mean brightness of that band
          3. Maintain a rolling 30-frame baseline (open-eye brightness)
          4. If current brightness drops > DROP_THRESHOLD below baseline
             → eye is closing → state = "closed"
          5. Otherwise → state = "open"

        This produces real open/closed transitions that _tick_blink can count.
        """
        DROP_THRESHOLD = 0.12   # 12% brightness drop triggers "closed"
        ALPHA          = 0.05   # baseline update speed (slow — only when open)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

            # Eye-region crop
            y1 = int(h * 0.20)
            y2 = int(h * 0.50)
            x1 = int(w * 0.25)
            x2 = int(w * 0.75)
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                return self._prev_eye_state, 0.50

            score = float(cv2.mean(roi)[0]) / 255.0
            self._collect_intensity_calibration(score)

            if self._calibrated:
                raw_state, raw_conf = self._brightness_from_calibration(score)
                self._update_adaptive_threshold(score, raw_state)
            else:
                if self._intensity_baseline < 0:
                    self._intensity_baseline = score
                self._intensity_history.append(score)
                drop = self._intensity_baseline - score
                if drop > DROP_THRESHOLD:
                    raw_state = "closed"
                    raw_conf = min(0.95, 0.55 + (drop - DROP_THRESHOLD) * 4.0)
                else:
                    raw_state = "open"
                    raw_conf = min(0.95, 0.55 + (DROP_THRESHOLD - drop) * 2.0)
                    self._intensity_baseline = (
                        (1.0 - ALPHA) * self._intensity_baseline + ALPHA * score
                    )

            state, conf = self._stabilize_eye_state(raw_state, raw_conf)
            return state, round(conf, 3)

        except Exception as e:
            logger.debug("[Vision] _intensity_blink_detect error: %s", e)
            return self._prev_eye_state, 0.50

    # ── Blink state machine ────────────────────────────────────────────────────

    def _tick_blink(self, current_state: str) -> None:
        """
        Counts a blink on CLOSED → OPEN transition.
        Matches the original inference.py logic exactly:
          prev=CLOSED, current=OPEN → one complete blink registered.
        """
        if self._last_known_state == "closed" and current_state == "open":
            self._blink_count += 1
            now_ts = time.monotonic()
            self._blink_times.append(now_ts)
            # Prune entries older than 60 seconds
            cutoff = now_ts - 60.0
            while self._blink_times and self._blink_times[0] < cutoff:
                self._blink_times.popleft()
            _db_log_blink(self._blink_count, self._compute_brpm_unsafe())
            logger.info("[Vision] BLINK ✓  Total=%d  BRPM=%.1f",
                        self._blink_count, self._compute_brpm_unsafe())

        self._last_known_state = current_state

    # ── Blink rate ─────────────────────────────────────────────────────────────

    def _compute_brpm(self) -> float:
        with self._blink_lock:
            return self._compute_brpm_unsafe()

    def _compute_brpm_unsafe(self) -> float:
        """Sliding 60-second window blink rate."""
        now = time.monotonic()
        cutoff = now - 60.0
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()

        n = len(self._blink_times)
        if n == 0:
            return 0.0

        # Use elapsed time since first blink in the window
        # Minimum 5 seconds to avoid absurdly high rates on first blink
        elapsed = max(now - self._blink_times[0], 5.0)
        return round(n / elapsed * 60.0, 1)

    # ── XGBoost ────────────────────────────────────────────────────────────────

    def _xgb_loop(self) -> None:
        """
        Dedicated background thread that runs XGBoost every PREDICTION_INTERVAL_S.

        Runs independently of the camera — so DED predictions work even when
        only MQTT sensors are connected (no ESP32-CAM).

        Uses the current blink rate from the vision pipeline if available,
        otherwise uses 0.0 (which triggers the heuristic fallback in ai_model).
        """
        logger.info("[Vision] XGBoost loop started (interval=%.1fs)", PREDICTION_INTERVAL_S)
        while not self._stop_evt.is_set():
            # Wait for the interval, checking stop every second
            for _ in range(int(PREDICTION_INTERVAL_S)):
                if self._stop_evt.is_set():
                    return
                time.sleep(1.0)

            # Get current blink rate (thread-safe)
            brpm = self._compute_brpm()
            self._run_xgboost(brpm)

    def _run_xgboost(self, brpm: float) -> None:
        try:
            from ai_model import predict
        except ImportError as e:
            logger.error("[Vision] ai_model import failed: %s", e)
            return

        snap    = get_snapshot()
        sensors = snap.get("sensors", {})
        status  = snap.get("status", {})

        temp     = sensors.get("temperature", 0.0)
        humidity = sensors.get("humidity",    0.0)
        lux      = sensors.get("lux",         0.0)
        eye_temp = sensors.get("eye_temp",    0.0)

        # Allow XGBoost to run when either sensors OR blink data is available.
        # This means predictions work with camera-only (no MQTT sensors).
        has_sensors = any(v > 0.0 for v in (temp, humidity, lux, eye_temp, brpm))

        if not has_sensors:
            # No real sensor data — keep prediction as "En attente"
            update_prediction(
                label="En attente",
                confidence=0.0,
                recommendation="Connectez les capteurs pour démarrer le diagnostic."
            )
            return

        features = {
            "temp":       _sensor_val(temp,     25.0),
            "humidity":   _sensor_val(humidity, 50.0),
            "lux":        _sensor_val(lux,      300.0),
            "eye_temp":   _sensor_val(eye_temp, 34.5),
            "blink_rate": brpm,
        }

        try:
            result = predict(features)
            try:
                update_prediction(
                    label=result.get("prediction", ""),
                    confidence=result.get("confidence", 0.0),
                    class_idx={"Sain": 0, "Risque Modéré": 1, "Risque Sévère": 2}.get(
                        result.get("prediction", ""), -1
                    ),
                    recommendation=result.get("recommendation", ""),
                )
            except Exception as _e:
                logger.warning("[vision_processor] state.update_prediction failed: %s", _e)
            vision = snap.get("vision", {})
            eye_state = vision.get("eye_state", "unknown")
            blink_count = int(vision.get("blink_count", 0))
            prediction_label = result.get("prediction", "")
            confidence = result.get("confidence", 0.0)
            recommendation = result.get("recommendation", "")

            _db_log_measurement(
                timestamp=status.get("last_update", "") or "",
                temperature=features["temp"],
                humidity=features["humidity"],
                lux=features["lux"],
                eye_temp=features["eye_temp"],
                blink_rate=brpm,
                blink_count=blink_count,
                eye_state=eye_state,
                prediction=prediction_label,
                confidence=confidence,
                recommendation=recommendation,
            )

            logger.info(
                "[Vision] XGBoost → %s (%.2f) | blink=%.1f/min "
                "temp=%.1f hum=%.1f lux=%.0f eye=%.1f",
                prediction_label, confidence, brpm,
                features["temp"] or 0.0, features["humidity"] or 0.0,
                features["lux"] or 0.0, features["eye_temp"] or 0.0,
            )
        except Exception as e:
            logger.error("[Vision] XGBoost error: %s", e)