"""
mqtt_client.py
────────────────────────────────────────────────────────────────
Wrapper MQTT pour DED-Monitor.

FIX: Utilisation de CallbackAPIVersion.VERSION1 (callbacks classiques)
     La VERSION2 de Paho v2 change le type du paramètre `rc` en objet
     ReasonCode — `if rc == 0:` peut échouer silencieusement selon la
     version installée, ce qui empêche la souscription au topic et
     bloque toute réception de messages capteurs.
"""

import json
import logging
import paho.mqtt.client as mqtt
from state import update_sensors, update_status

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_first_key(data: dict, keys: list):
    for key in keys:
        if key in data:
            return data.get(key)
    return None


def _normalize_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", ".")
        if not cleaned or cleaned.lower() in ("null", "none", "nan", "n/a"):
            return None
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"⚠️ Valeur capteur invalide ignorée: {value!r}")
            return None
    logger.warning(f"⚠️ Type de capteur inattendu: {type(value).__name__} {value!r}")
    return None


# ── Client MQTT ───────────────────────────────────────────────────────────────

class MQTTClient:
    """
    Wrapper MQTT compatible Paho v1 et v2.
    Utilise CallbackAPIVersion.VERSION1 pour des callbacks stables
    (rc est un entier, souscription garantie dans on_connect).
    """

    def __init__(self, host="172.20.10.4", port=1883):
        self.host  = host
        self.port  = port
        self.topic = "ded/sensors"

        # VERSION1 : signatures classiques, rc = entier, fiable sur toutes versions Paho 2.x
        import os as _os
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
            client_id=f"DED_Monitor_Backend_{_os.getpid()}",
            clean_session=True,
        )

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc):
        """Callback VERSION1 : rc est un entier (0 = succès)."""
        if rc == 0:
            logger.info(f"✅ Connecté au broker MQTT {self.host}:{self.port}")
            client.subscribe(self.topic, qos=0)
            logger.info(f"📡 Souscrit au topic: {self.topic}")
            update_status(mqtt_connected=True)
        else:
            reason = {
                1: "Protocole refusé",
                2: "Identifiant refusé",
                3: "Broker indisponible",
                4: "Identifiants invalides",
                5: "Non autorisé",
            }.get(rc, f"Code {rc}")
            logger.error(f"❌ Échec connexion MQTT — {reason}")
            update_status(mqtt_connected=False)

    def _on_disconnect(self, client, userdata, rc):
        """Callback VERSION1 : rc entier, 0 = déconnexion propre."""
        if rc == 0:
            logger.info("🔌 Déconnexion MQTT propre.")
        else:
            logger.warning(f"⚠️ Déconnexion MQTT inattendue (rc={rc}) — reconnexion automatique…")
        update_status(mqtt_connected=False)

    def _on_message(self, client, userdata, msg):
        raw = None
        try:
            raw = msg.payload.decode("utf-8")
            logger.info(f"[MQTT] Message received on '{msg.topic}': {raw[:100]}")
            data = json.loads(raw)

            temperature = _normalize_numeric(_find_first_key(data, [
                "temperature", "temp", "ambient", "temp_c", "ambient_temp", "air_temp",
            ]))
            humidity = _normalize_numeric(_find_first_key(data, [
                "humidity", "hum", "humidity_pct", "humidity_percent",
            ]))
            lux = _normalize_numeric(_find_first_key(data, [
                "lux", "light", "illumination", "illuminance",
            ]))
            eye_temp = _normalize_numeric(_find_first_key(data, [
                "eye_temp", "temp_eye", "eyeTemperature", "eye_temperature",
                "eye_temp_c", "eyeTemp",
            ]))

            logger.info(
                "📩 MQTT capteurs | temp=%.1f  hum=%.1f  lux=%.0f  eye_temp=%.1f",
                temperature or 0, humidity or 0, lux or 0, eye_temp or 0,
            )

            if all(v is None for v in (temperature, humidity, lux, eye_temp)):
                logger.warning(
                    "⚠️ Aucune clé capteur reconnue dans le payload MQTT. "
                    "Clés reçues: %s", list(data.keys())
                )
                return

            update_sensors(
                temperature=temperature,
                humidity=humidity,
                lux=lux,
                eye_temp=eye_temp,
            )

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON invalide dans le payload MQTT: {e} — brut='{raw}'")
        except Exception as e:
            logger.exception(f"❌ Erreur on_message: {e}")

    # ── Cycle de vie ──────────────────────────────────────────────────────────

    def start(self):
        """Démarre la connexion et la boucle réseau en arrière-plan."""
        self._client.reconnect_delay_set(min_delay=2, max_delay=30)
        self._client.loop_start()
        # Run the initial connection attempt in a background thread
        # so FastAPI startup is not blocked
        import threading
        threading.Thread(target=self._connect_with_retry, daemon=True,
                         name="MQTTConnect").start()

    def _connect_with_retry(self):
        """Tries to connect to the broker, retrying every 5s until success."""
        import time
        while True:
            try:
                logger.info(f"🔄 Connexion MQTT → {self.host}:{self.port}  topic={self.topic}")
                self._client.connect(self.host, self.port, keepalive=60)
                return  # Success — Paho loop_start handles the rest
            except (ConnectionRefusedError, OSError) as e:
                logger.warning(
                    f"⚠️ MQTT connexion échouée ({e}) — nouvelle tentative dans 5s…"
                )
                time.sleep(5)
            except Exception as e:
                logger.error(f"❌ Erreur MQTT inattendue: {e}")
                return

    def stop(self):
        """Arrêt propre."""
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass
        logger.info("🔌 MQTT arrêté.")

    # ── Command publisher ──────────────────────────────────────────────────────────

    COMMAND_TOPIC = "ded/commands"

    def publish_command(self, severity: str) -> bool:
        """
        Publishes an alert command to the ESP32 via MQTT.

        severity : "SEVERE"   → buzzer + vibrator ON
                   "MODERATE" → vibrator ON, buzzer OFF
                   "NORMAL"   → both OFF
        Returns True if published successfully.
        """
        valid = {"SEVERE", "MODERATE", "NORMAL"}
        if severity not in valid:
            logger.warning("[MQTT] publish_command: invalid severity '%s'", severity)
            return False
        try:
            result = self._client.publish(self.COMMAND_TOPIC, severity, qos=1)

            publish_ok = False
            if hasattr(result, "rc") and result.rc == 0:
                publish_ok = True
            elif hasattr(result, "is_published") and result.is_published():
                publish_ok = True
            elif result == 0:
                publish_ok = True

            if publish_ok:
                logger.info("[MQTT] ✅ Command published → %s", severity)
                return True

            logger.warning("[MQTT] ⚠️ Command publish failed result=%s", result)
            return False
        except Exception as e:
            logger.error("[MQTT] ❌ publish_command error: %s", e)
            return False