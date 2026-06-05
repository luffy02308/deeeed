"""
web/backend/ai_model.py
────────────────────────────────────────────────────────────────────────────────
Wrapper de prédiction DED — utilise ml_loader (singleton) pour le modèle
XGBoost + StandardScaler, avec les 7 features dans l'ordre exact
d'entraînement : [blink_rate, humidity, temperature, eye_temp, lux, age, sexe]

Interface publique :
    predict(features_dict) → dict(prediction, confidence, status, recommendation)

Compatibilité ascendante :
    - La clé 'temp' est acceptée comme alias de 'temperature'.
    - Les clés manquantes sont imputées avec les valeurs cliniques par défaut.
    - Si ml_loader n'est pas encore initialisé (appel trop précoce au démarrage),
      un fallback heuristique est utilisé et loggé en WARNING.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Valeurs par défaut cliniquement neutres ───────────────────────────────────
# Identiques à predict_router._FEATURE_DEFAULTS et train_ded_xgboost._DEFAULTS.
_FEATURE_DEFAULTS: Dict[str, float] = {
    "blink_rate":  14.0,   # Ousler et al. (2007) : groupe normal ≈ 13.2 /min
    "humidity":    55.0,   # Valeur de bureau neutre
    "temperature": 23.0,   # ASHRAE 55 : plage de confort
    "eye_temp":    34.0,   # Ramlan et al. (2022) : groupe normal ≈ 33.9 °C
    "lux":        300.0,   # Bureau standard
    "age":         30.0,   # Médiane des cohortes
    "sexe":         0.5,   # Neutre (genre inconnu) — compatible avec le scaler
}

# ── Labels et recommandations (identiques à ml_loader.LABEL_MAP) ─────────────
_LABEL_MAP: Dict[int, str] = {
    0: "Sain",
    1: "Risque Modéré",
    2: "Risque Sévère",
}

_RECOMMANDATIONS: Dict[int, str] = {
    0: "État normal. Maintenez des pauses régulières toutes les 20 minutes (règle 20-20-20).",
    1: "Clignez plus souvent. Utilisez des larmes artificielles. Réduisez l'exposition aux écrans.",
    2: "⚠️ Sécheresse oculaire sévère. Consultez un ophtalmologue rapidement.",
}

_LEGACY_FEATURE_ORDER: list[str] = [
    "blink_rate",
    "humidity",
    "temp_ambient",
    "luminosity",
    "temp_eye",
    "temp_diff",
]


def _build_legacy_features(raw: Dict[str, Any]) -> dict[str, float]:
    """Build legacy XGBoost inputs for older scaler/model assets."""
    temp_ambient = raw.get("temperature", raw.get("temp_ambient", _FEATURE_DEFAULTS["temperature"]))
    eye_temp = raw.get("eye_temp", raw.get("temp_eye", _FEATURE_DEFAULTS["eye_temp"]))
    luminosity = raw.get("lux", raw.get("luminosity", _FEATURE_DEFAULTS["lux"]))
    humidity = raw.get("humidity", _FEATURE_DEFAULTS["humidity"])

    try:
        temp_ambient = float(temp_ambient)
    except (TypeError, ValueError):
        temp_ambient = _FEATURE_DEFAULTS["temperature"]
    try:
        eye_temp = float(eye_temp)
    except (TypeError, ValueError):
        eye_temp = _FEATURE_DEFAULTS["eye_temp"]
    try:
        luminosity = float(luminosity)
    except (TypeError, ValueError):
        luminosity = _FEATURE_DEFAULTS["lux"]
    try:
        humidity = float(humidity)
    except (TypeError, ValueError):
        humidity = _FEATURE_DEFAULTS["humidity"]

    temp_diff = eye_temp - temp_ambient
    return {
        "blink_rate": float(raw.get("blink_rate", _FEATURE_DEFAULTS["blink_rate"])),
        "humidity": humidity,
        "temp_ambient": temp_ambient,
        "luminosity": luminosity,
        "temp_eye": eye_temp,
        "temp_diff": temp_diff,
    }


# ── API publique ──────────────────────────────────────────────────────────────

def predict(features_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prédit la classe DED à partir des données capteurs + CNN.

    Paramètres
    ----------
    features_dict : dict contenant tout ou partie des clés :
        blink_rate  — fréquence de clignement (CNN, /min)
        humidity    — humidité ambiante (DHT22, %)
        temperature — température ambiante (DHT22, °C)   [alias : 'temp']
        eye_temp    — température oculaire (MLX90614, °C)
        lux         — luminosité ambiante (BH1750, lux)
        age         — âge du patient (années)            [défaut : 30.0]
        sexe        — 0=Homme, 1=Femme                   [défaut : 0.5]

    Retour
    ------
    dict avec : prediction, confidence, class_idx, status, recommendation, features_used
    """
    raw: Dict[str, Any] = dict(features_dict)
    if "temp" in raw and "temperature" not in raw:
        raw["temperature"] = raw.pop("temp")
    if "temp_ambient" in raw and "temperature" not in raw:
        raw["temperature"] = raw["temp_ambient"]
    if "lux" not in raw and "luminosity" in raw:
        raw["lux"] = raw["luminosity"]
    if "eye_temp" not in raw and "temp_eye" in raw:
        raw["eye_temp"] = raw["temp_eye"]

    try:
        import ml_loader
        feature_order = ml_loader.FEATURE_ORDER
    except Exception:
        feature_order = list(_FEATURE_DEFAULTS.keys())

    imputed: Dict[str, float] = {}
    missing: list[str] = []
    for feat in feature_order:
        val = raw.get(feat)
        if val is None:
            imputed[feat] = _FEATURE_DEFAULTS[feat]
            missing.append(feat)
        else:
            try:
                imputed[feat] = float(val)
            except (TypeError, ValueError):
                imputed[feat] = _FEATURE_DEFAULTS[feat]
                missing.append(feat)

    if missing:
        logger.debug("[ai_model] Champs manquants → valeurs par défaut : %s", missing)

    features_used = {feat: imputed[feat] for feat in feature_order}

    try:
        import ml_loader
        xgb_model, scaler = ml_loader.get_assets()
        feature_order = ml_loader.FEATURE_ORDER
    except RuntimeError:
        logger.warning(
            "[ai_model] ml_loader non initialisé — fallback heuristique. "
            "Vérifiez que load_assets() est bien appelé dans le lifespan FastAPI."
        )
        return _heuristic_fallback(raw, features_used)
    except Exception as e:
        logger.error("[ai_model] Erreur accès ml_loader : %s", e)
        return _heuristic_fallback(raw, features_used)

    scaler_expected = getattr(scaler, "n_features_in_", None)
    if scaler_expected == len(feature_order):
        X_input = [[imputed[feat] for feat in feature_order]]
    elif scaler_expected == len(_LEGACY_FEATURE_ORDER):
        legacy_inputs = _build_legacy_features(raw)
        X_input = [[legacy_inputs[feat] for feat in _LEGACY_FEATURE_ORDER]]
    else:
        # Fallback to the number of features the scaler actually expects.
        fallback_order = feature_order[: scaler_expected] if scaler_expected else feature_order
        X_input = [[imputed[feat] for feat in fallback_order]]

    X_input_arr = np.asarray(X_input, dtype=float)

    try:
        X_scaled  = scaler.transform(X_input_arr)
        pred_idx  = int(xgb_model.predict(X_scaled)[0])
        proba_vec = xgb_model.predict_proba(X_scaled)[0]
        confidence = float(proba_vec[pred_idx])
    except Exception as e:
        logger.error("[ai_model] Erreur pipeline ML : %s", e)
        return _heuristic_fallback(raw, features_used)

    label = _LABEL_MAP.get(pred_idx, f"Classe {pred_idx}")
    logger.info(
        "[ai_model] ✅ %s (%.3f) | blink=%.1f  eye_temp=%.1f  hum=%.1f  lux=%.0f",
        label, confidence,
        imputed["blink_rate"], imputed["eye_temp"],
        imputed["humidity"],   imputed["lux"],
    )

    try:
        from state import update_prediction
        update_prediction(
            label=label,
            confidence=round(confidence, 4),
            class_idx=pred_idx,
            recommendation=_RECOMMANDATIONS.get(pred_idx, ""),
        )
    except Exception as _e:
        logger.warning("[ai_model] state.update_prediction failed: %s", _e)

    # ── Publish ESP32 alert command based on prediction ───────────────────────
    try:
        from main import _mqtt_client
        _SEVERITY_MAP: dict[int, str] = {0: "NORMAL", 1: "MODERATE", 2: "SEVERE"}
        _mqtt_client.publish_command(_SEVERITY_MAP.get(pred_idx, "NORMAL"))
    except Exception as _cmd_err:
        logger.warning("[ai_model] Could not publish ESP32 command: %s", _cmd_err)

    return {
        "prediction":     label,
        "confidence":     round(confidence, 4),
        "class_idx":      pred_idx,
        "status":         "success",
        "recommendation": _RECOMMANDATIONS.get(pred_idx, ""),
        "features_used":  features_used,
    }


# ── Fallback heuristique (modèle non disponible) ─────────────────────────

def _heuristic_fallback(
    features_dict: Dict[str, Any],
    features_used: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Classification heuristique basée sur blink_rate et eye_temp uniquement.
    Utilisé quand ml_loader n'est pas encore initialisé ou a échoué.
    """
    try:
        blink_rate = float(
            features_dict.get("blink_rate")
            or features_dict.get("temp")
            or _FEATURE_DEFAULTS["blink_rate"]
        )
    except (TypeError, ValueError):
        blink_rate = _FEATURE_DEFAULTS["blink_rate"]

    try:
        eye_temp = float(
            features_dict.get("eye_temp") or _FEATURE_DEFAULTS["eye_temp"]
        )
    except (TypeError, ValueError):
        eye_temp = _FEATURE_DEFAULTS["eye_temp"]

    if features_used is None:
        features_used = {}

    if blink_rate < 7.0 or eye_temp < 33.0:
        idx = 2   # Sévère
    elif blink_rate < 12.0 or eye_temp < 33.7:
        idx = 1   # Modéré
    else:
        idx = 0   # Normal

    logger.warning(
        "[ai_model] Mode heuristique → %s | blink=%.1f  eye_temp=%.1f",
        _LABEL_MAP[idx], blink_rate, eye_temp,
    )

    try:
        from main import _mqtt_client
        _SEVERITY_MAP: dict[int, str] = {0: "NORMAL", 1: "MODERATE", 2: "SEVERE"}
        _mqtt_client.publish_command(_SEVERITY_MAP.get(idx, "NORMAL"))
    except Exception as _cmd_err:
        logger.warning("[ai_model] Could not publish fallback ESP32 command: %s", _cmd_err)

    return {
        "prediction":     _LABEL_MAP[idx],
        "confidence":     0.70,
        "class_idx":      idx,
        "status":         "simulated",
        "recommendation": _RECOMMANDATIONS[idx],
        "features_used":  features_used,
    }
