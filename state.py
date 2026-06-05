"""
services/state.py
─────────────────────────────────────────────────────────────────
Gestionnaire d'état global optimisé pour DED-Monitor.
"""

from __future__ import annotations
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Set, Dict, Any

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODÈLES DE DONNÉES (STRUCTURES)
# ════════════════════════════════════════════════───────────────────────────────

@dataclass
class SensorValues:
    temperature: Optional[float] = None  
    humidity:    Optional[float] = None  
    lux:         Optional[float] = None  
    eye_temp:    Optional[float] = None  

@dataclass
class VisionValues:
    eye_state:             str   = "unknown"
    eye_confidence:        float = 0.0
    blink_count:           int   = 0
    blink_rate_per_minute: float = 0.0
    ear:                   float = -1.0

@dataclass
class PredictionValues:
    label:          str   = "En attente"
    confidence:     float = 0.0
    class_idx:      int   = -1
    recommendation: str   = ""  # Ajouté pour le frontend

@dataclass
class StatusValues:
    camera_online:  bool = False
    mqtt_connected: bool = False
    last_update:    str  = ""

@dataclass
class AppState:
    sensors:    SensorValues    = field(default_factory=SensorValues)
    vision:     VisionValues    = field(default_factory=VisionValues)
    prediction: PredictionValues = field(default_factory=PredictionValues)
    status:     StatusValues    = field(default_factory=StatusValues)

# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLETONS & BROADCASTER
# ═══════════════════════════════════════════════════════════════════════════════

_app_state  = AppState()
_state_lock = threading.Lock()

class Broadcaster:
    def __init__(self) -> None:
        self._queues: Set[asyncio.Queue] = set()
        self._set_lock: threading.Lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def add_subscriber(self, q: asyncio.Queue) -> None:
        with self._set_lock: self._queues.add(q)

    def remove_subscriber(self, q: asyncio.Queue) -> None:
        with self._set_lock: self._queues.discard(q)

    def publish(self, payload: dict) -> None:
        if self._loop is None or self._loop.is_closed(): return
        with self._set_lock:
            for q in list(self._queues):
                try:
                    asyncio.run_coroutine_threadsafe(self._safe_put(q, payload), self._loop)
                except Exception: pass

    @staticmethod
    async def _safe_put(q: asyncio.Queue, item: dict) -> None:
        try:
            if q.full(): q.get_nowait()
            q.put_nowait(item)
        except Exception: pass

broadcaster = Broadcaster()

# ═══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS PUBLIQUES (API INTERNE)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"⚠️ Valeur de capteur non numérique ignorée: {value!r}")
        return None


def update_sensors(temperature=None, humidity=None, lux=None, eye_temp=None):
    with _state_lock:
        s = _app_state.sensors
        if temperature is not None:
            temp_val = _parse_float(temperature)
            if temp_val is not None:
                s.temperature = temp_val
        if humidity is not None:
            hum_val = _parse_float(humidity)
            if hum_val is not None:
                s.humidity = hum_val
        if lux is not None:
            lux_val = _parse_float(lux)
            if lux_val is not None:
                s.lux = lux_val
        if eye_temp is not None:
            eye_temp_val = _parse_float(eye_temp)
            if eye_temp_val is not None:
                s.eye_temp = eye_temp_val
        _finish_update()

def update_vision(eye_state=None, eye_confidence=None, blink_count=None, blink_rate=None, ear=None):
    with _state_lock:
        v = _app_state.vision
        if eye_state      is not None: v.eye_state      = str(eye_state)
        if eye_confidence is not None: v.eye_confidence = float(eye_confidence)
        if blink_count    is not None: v.blink_count    = int(blink_count)
        if blink_rate     is not None: v.blink_rate_per_minute = float(blink_rate)
        if ear            is not None: v.ear            = float(ear)
        _finish_update()

def update_prediction(label=None, confidence=None, class_idx=None, recommendation=None):
    """Met à jour les résultats de l'IA XGBoost."""
    with _state_lock:
        p = _app_state.prediction
        if label          is not None: p.label          = str(label)
        if confidence     is not None: p.confidence     = float(confidence)
        if class_idx      is not None: p.class_idx      = int(class_idx)
        if recommendation is not None: p.recommendation = str(recommendation)
        _finish_update()

def update_status(camera_online=None, mqtt_connected=None):
    with _state_lock:
        s = _app_state.status
        if camera_online  is not None: s.camera_online  = bool(camera_online)
        if mqtt_connected is not None: s.mqtt_connected = bool(mqtt_connected)
        _finish_update()

def get_snapshot() -> dict:
    with _state_lock:
        return _build_snapshot_dict()

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS INTERNES
# ═══════════════════════════════════════════════════════════════════════════════

def _finish_update():
    """Marque le temps et notifie les abonnés."""
    _app_state.status.last_update = datetime.now(timezone.utc).isoformat()
    broadcaster.publish(_build_snapshot_dict())

def _build_snapshot_dict() -> dict:
    s, v, p, st = _app_state.sensors, _app_state.vision, _app_state.prediction, _app_state.status
    return {
        "type": "state_update",
        "sensors": {
            "temperature": 0.0 if s.temperature is None else s.temperature,
            "humidity":    0.0 if s.humidity is None else s.humidity,
            "lux":         0.0 if s.lux is None else s.lux,
            "eye_temp":    0.0 if s.eye_temp is None else s.eye_temp,
        },
        "vision": {
            "eye_state": v.eye_state,
            "eye_confidence": round(v.eye_confidence, 4),
            "blink_count": v.blink_count,
            "blink_rate_per_minute": round(v.blink_rate_per_minute, 2),
            "ear": round(v.ear, 4)
        },
        "prediction": {
            "label": p.label,
            "confidence": round(p.confidence, 4),
            "recommendation": p.recommendation
        },
        "status": {
            "camera_online": st.camera_online,
            "mqtt_connected": st.mqtt_connected,
            "last_update": st.last_update
        }
    }