"""
web/backend/predict_router.py
────────────────────────────────────────────────────────────────────────────────
Routeur FastAPI pour l'endpoint de prédiction DED multiclasse.

Ce module est autonome : il ne sait rien de main.py, il importe uniquement
ml_loader (pour les assets IA) et expose un APIRouter prêt à être monté.

Intégration dans main.py :
    from predict_router import router as predict_router
    app.include_router(predict_router)

Endpoint exposé :
    POST /api/predict
    → Retourne prediction_code, eye_state, confidence_score
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Prédiction DED"])


# ══════════════════════════════════════════════════════════════════════════════
#  SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    """
    Payload entrant pour une prédiction DED.

    Tous les champs sont optionnels : si une valeur est absente, la valeur
    par défaut cliniquement neutre est utilisée (et loguée en WARNING).
    Cela permet d'appeler l'endpoint depuis l'ESP32 même si un capteur
    est temporairement défaillant.

    Ordre d'entraînement : blink_rate, humidity, temperature, eye_temp,
                           lux, age, sexe   ← NE PAS MODIFIER
    """

    blink_rate:  Optional[float] = Field(
        default=None,
        description="Fréquence de clignement (clignements/min) — CNN sur flux caméra",
        ge=0.0, le=60.0,
    )
    humidity:    Optional[float] = Field(
        default=None,
        description="Humidité relative ambiante (%) — DHT22",
        ge=0.0, le=100.0,
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Température ambiante (°C) — DHT22",
        ge=-20.0, le=60.0,
    )
    temp: Optional[float] = Field(
        default=None,
        description="Alias de temperature — compatibilité ascendante",
        ge=-20.0, le=60.0,
    )
    eye_temp:    Optional[float] = Field(
        default=None,
        description="Température de surface oculaire (°C) — MLX90614",
        ge=20.0, le=42.0,
    )
    lux:         Optional[float] = Field(
        default=None,
        description="Luminosité ambiante (lux) — BH1750",
        ge=0.0,
    )
    age:         Optional[float] = Field(
        default=None,
        description="Âge du patient (années)",
        ge=0.0, le=120.0,
    )
    sexe:        Optional[float] = Field(
        default=None,
        description="Sexe biologique : 0 = Homme, 1 = Femme",
    )

    @field_validator("sexe")
    @classmethod
    def validate_sexe(cls, v: Optional[float]) -> Optional[float]:
        """Accepte uniquement 0.0 et 1.0 (ou None)."""
        if v is not None and v not in (0.0, 1.0):
            raise ValueError("sexe doit être 0 (Homme) ou 1 (Femme)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "blink_rate":  14.5,
                "humidity":    55.0,
                "temperature": 23.5,
                "eye_temp":    34.2,
                "lux":         320.0,
                "age":         28.0,
                "sexe":        1.0,
            }
        }
    }


class PredictResponse(BaseModel):
    """Réponse structurée retournée par POST /api/predict."""

    prediction_code:  int   = Field(description="Code entier : 0=Sain, 1=Risque Modéré, 2=Risque Sévère")
    eye_state:        str   = Field(description="Libellé humain de la prédiction")
    confidence_score: float = Field(description="Probabilité brute de la classe prédite (0.0–1.0)")
    features_used:    dict  = Field(description="Valeurs réelles envoyées au modèle (après imputation)")


class CommandRequest(BaseModel):
    """Payload de commande manuelle vers l'ESP32."""

    severity: Literal["NORMAL", "MODERATE", "SEVERE"] = Field(
        description="Niveau d'alerte envoyé à l'ESP32."
    )


class CommandResponse(BaseModel):
    """Réponse de l'endpoint POST /api/command."""

    status: str = Field(description="Status de l'envoi de commande")
    command: str = Field(description="Commande MQTT envoyée")


# ── Valeurs par défaut pour les champs manquants ───────────────────────────────
# Basées sur les moyennes des groupes "normaux" extraits des articles cliniques.
_FEATURE_DEFAULTS: dict[str, float] = {
    "blink_rate":  14.0,   # Art. 3 (Ousler) : normal group mean ≈ 13.2 blinks/min
    "humidity":    55.0,   # Valeur neutre typique de bureau
    "temperature": 23.0,   # Plage de confort bureau (ASHRAE 55)
    "eye_temp":    34.0,   # Art. 1 (Ramlan) : groupe normal ≈ 33.9 °C
    "lux":        300.0,   # Art. 4 (Zheng) : 100 lux labo ; 300 lux bureau standard
    "age":         30.0,   # Valeur médiane des cohortes étudiées
    "sexe":         0.5,   # Valeur neutre (genre inconnu) — compatible avec scaler
}

_RECOMMANDATIONS: dict[int, str] = {
    0: "État normal. Maintenez des pauses régulières toutes les 20 minutes (règle 20-20-20).",
    1: "Clignez plus souvent. Utilisez des larmes artificielles. Réduisez l'exposition aux écrans.",
    2: "⚠️ Sécheresse oculaire sévère. Consultez un ophtalmologue rapidement.",
}


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédiction DED multiclasse",
    description=(
        "Reçoit les données capteurs et retourne la classification DED "
        "(Sain / Risque Modéré / Risque Sévère) avec son score de confiance."
    ),
    status_code=status.HTTP_200_OK,
)
async def predict_ded(payload: PredictRequest) -> PredictResponse:
    """
    Pipeline de prédiction :
        1. Imputer les valeurs manquantes avec les défauts cliniques.
        2. Construire un DataFrame Pandas dans l'ordre exact d'entraînement.
        3. Normaliser avec le StandardScaler chargé au démarrage.
        4. Prédire avec le modèle XGBoost chargé au démarrage.
        5. Retourner prediction_code + eye_state + confidence_score.
    """

    from ai_model import predict as ai_predict
    from state import update_prediction
    import ml_loader

    raw = payload.model_dump()

    result = ai_predict(raw)

    if result.get("status") == "success":
        try:
            xgb_model, scaler = ml_loader.get_assets()
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Le modèle IA n'est pas chargé.",
            )

    _RECOMMANDATIONS = {
        0: "État normal. Maintenez des pauses régulières toutes les 20 minutes.",
        1: "Clignez plus souvent. Utilisez des larmes artificielles.",
        2: "⚠️ Sécheresse oculaire sévère. Consultez un ophtalmologue rapidement.",
    }
    _LABEL_TO_CODE = {"Sain": 0, "Risque Modéré": 1, "Risque Sévère": 2}

    label = result["prediction"]
    confidence = result["confidence"]
    pred_code = _LABEL_TO_CODE.get(label, -1)

    update_prediction(
        label=label,
        confidence=confidence,
        class_idx=pred_code,
        recommendation=result.get("recommendation", ""),
    )

    logger.info(
        "[Predict] ✅ code=%d  état='%s'  confiance=%.4f",
        pred_code, label, confidence,
    )

    features_used = result.get("features_used")
    if not isinstance(features_used, dict):
        features_used = {}
        for feat in ml_loader.FEATURE_ORDER:
            val = raw.get(feat)
            features_used[feat] = float(val) if val is not None else {
                "blink_rate": 14.0, "humidity": 55.0, "temperature": 23.0,
                "eye_temp": 34.0, "lux": 300.0, "age": 30.0, "sexe": 0.5,
            }[feat]

    return PredictResponse(
        prediction_code=pred_code,
        eye_state=label,
        confidence_score=round(confidence, 6),
        features_used=features_used,
    )


@router.post(
    "/command",
    response_model=CommandResponse,
    summary="Envoi de commande manuelle à l'ESP32",
    description=(
        "Reçoit une commande manuelle et la publie sur le topic MQTT de l'ESP32."
    ),
    status_code=status.HTTP_200_OK,
)
async def send_command(payload: CommandRequest) -> CommandResponse:
    try:
        from main import _mqtt_client
        published = _mqtt_client.publish_command(payload.severity)
    except Exception as exc:
        logger.error("[PredictRouter] Error publishing manual command: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Impossible de publier la commande MQTT.",
        )

    if not published:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Publication MQTT échouée.",
        )

    return CommandResponse(status="success", command=payload.severity)