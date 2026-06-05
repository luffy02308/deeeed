"""
web/backend/ml_loader.py
────────────────────────────────────────────────────────────────────────────────
Chargement unique (singleton) du modèle XGBoost + StandardScaler pour DED-Monitor.

Responsabilités :
  - Résoudre les chemins absolus vers ml_assets/ depuis ce fichier (jamais
    de chemins relatifs au répertoire de lancement du serveur).
  - Charger modele_ded_xgboost.pkl et scaler_ded.pkl en mémoire UNE SEULE FOIS
    au démarrage (appelé depuis le lifespan FastAPI).
  - Exposer load_assets() et get_assets() pour le reste du backend.
  - Lever des erreurs explicites si les fichiers sont absents ou corrompus.

Arborescence attendue :
    <project_root>/
    ├── ml_assets/
    │   ├── modele_ded_xgboost.pkl
    │   └── scaler_ded.pkl
    └── web/
        └── backend/
            └── ml_loader.py   ← ce fichier
"""

from __future__ import annotations

import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Résolution des chemins ────────────────────────────────────────────────────
# __file__ est toujours le chemin absolu de CE script (indépendant du cwd).
# On remonte de web/backend/ → web/ → project_root/ → ml_assets/
_THIS_FILE   = Path(os.path.abspath(__file__))          # .../web/backend/ml_loader.py
_BACKEND_DIR = _THIS_FILE.parent                        # .../web/backend/
_WEB_DIR     = _BACKEND_DIR.parent                      # .../web/
_PROJECT_ROOT = _WEB_DIR.parent                         # .../  (racine du projet)

ML_ASSETS_DIR          = _PROJECT_ROOT / "ml_assets"
BACKEND_ASSETS_DIR     = _BACKEND_DIR
MODEL_PATH             = ML_ASSETS_DIR / "modele_ded_xgboost.pkl"
SCALER_PATH            = ML_ASSETS_DIR / "scaler_ded.pkl"
FALLBACK_MODEL         = BACKEND_ASSETS_DIR / "modele_ded_xgboost.pkl"
FALLBACK_SCALER        = BACKEND_ASSETS_DIR / "scaler_ded.pkl"
BACKEND_AI_XGBOOST_DIR = _PROJECT_ROOT / "backend" / "ai_models" / "ai_models" / "xgboost"
SECONDARY_MODEL        = BACKEND_AI_XGBOOST_DIR / "model.pkl"
SECONDARY_SCALER       = BACKEND_AI_XGBOOST_DIR / "scaler.pkl"

# ── Constantes métier ─────────────────────────────────────────────────────────
# Ordre EXACT des colonnes attendu par le scaler et le modèle.
# Ne pas réordonner — un décalage silencieux fausserait toutes les prédictions.
FEATURE_ORDER: list[str] = [
    "blink_rate",   # Clignements/min — CNN sur flux ESP32-CAM
    "humidity",     # Humidité relative % — DHT22
    "temperature",  # Température ambiante °C — DHT22
    "eye_temp",     # Température oculaire °C — MLX90614
    "lux",          # Luminosité ambiante lux — BH1750
    "age",          # Âge du patient (années)
    "sexe",         # Sexe biologique : 0 = Homme, 1 = Femme
]

# Mapping code → étiquette humaine (correspondant aux classes d'entraînement)
LABEL_MAP: dict[int, str] = {
    0: "Sain",
    1: "Risque Modéré",
    2: "Risque Sévère",
}

# ── Singletons en mémoire ─────────────────────────────────────────────────────
_xgb_model: Optional[Any]  = None   # XGBoostClassifier
_scaler:    Optional[Any]  = None   # StandardScaler


def load_assets() -> None:
    """
    Charge modele_ded_xgboost.pkl et scaler_ded.pkl en mémoire.

    Doit être appelé UNE SEULE FOIS depuis le lifespan FastAPI au démarrage.
    Lève une RuntimeError si l'un des fichiers est introuvable ou illisible,
    afin d'échouer tôt (fail-fast) plutôt que silencieusement lors des requêtes.

    Exemple d'intégration dans main.py :
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            ml_loader.load_assets()   # ← ici
            yield
            ...
    """
    global _xgb_model, _scaler

    logger.info("[ML] Répertoire ml_assets/ : %s", ML_ASSETS_DIR)
    logger.info("[ML] Répertoire fallback backend/  : %s", BACKEND_ASSETS_DIR)

    # ── Vérification de l'existence des fichiers, avec fallback backend/ et backend/ai_models/...
    model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL
    scaler_path = SCALER_PATH if SCALER_PATH.exists() else FALLBACK_SCALER

    if not model_path.exists() and SECONDARY_MODEL.exists():
        logger.warning(
            "[ML] modèle ml_assets/ introuvable, utilisation de backend/ai_models/.../model.pkl"
        )
        model_path = SECONDARY_MODEL

    if not scaler_path.exists() and SECONDARY_SCALER.exists():
        logger.warning(
            "[ML] scaler ml_assets/ introuvable, utilisation de backend/ai_models/.../scaler.pkl"
        )
        scaler_path = SECONDARY_SCALER

    candidate_model_paths = [MODEL_PATH]
    candidate_scaler_paths = [SCALER_PATH]
    if FALLBACK_MODEL.exists():
        candidate_model_paths.append(FALLBACK_MODEL)
    if SECONDARY_MODEL.exists():
        candidate_model_paths.append(SECONDARY_MODEL)
    if FALLBACK_SCALER.exists():
        candidate_scaler_paths.append(FALLBACK_SCALER)
    if SECONDARY_SCALER.exists():
        candidate_scaler_paths.append(SECONDARY_SCALER)

    model_path = None
    scaler_path = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # Ignore: InconsistentVersionWarning

        for path in candidate_model_paths:
            if not path.exists():
                continue
            try:
                with open(path, "rb") as fh:
                    _xgb_model = pickle.load(fh)
                model_path = path
                logger.info("[ML] ✅ XGBoost chargé depuis %s", path)
                break
            except Exception as exc:
                logger.warning(
                    "[ML] échec chargement XGBoost depuis %s : %s",
                    path, exc,
                )

        if model_path is None:
            msg = (
                "[ML] ❌ Aucun modèle XGBoost valide trouvé dans :\n"
                + "\n".join(str(p) for p in candidate_model_paths)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        for path in candidate_scaler_paths:
            if not path.exists():
                continue
            try:
                with open(path, "rb") as fh:
                    _scaler = pickle.load(fh)
                scaler_path = path
                logger.info("[ML] ✅ Scaler chargé depuis %s", path)
                break
            except Exception as exc:
                logger.warning(
                    "[ML] échec chargement Scaler depuis %s : %s",
                    path, exc,
                )

        if scaler_path is None:
            msg = (
                "[ML] ❌ Aucun scaler valide trouvé dans :\n"
                + "\n".join(str(p) for p in candidate_scaler_paths)
            )
            logger.error(msg)
            raise RuntimeError(msg)


def get_assets() -> Tuple[Any, Any]:
    """
    Retourne (xgb_model, scaler) déjà chargés.

    Lève une RuntimeError si load_assets() n'a pas été appelé au préalable.
    """
    if _xgb_model is None or _scaler is None:
        raise RuntimeError(
            "[ML] get_assets() appelé avant load_assets(). "
            "Ajoutez ml_loader.load_assets() dans le lifespan FastAPI."
        )
    return _xgb_model, _scaler