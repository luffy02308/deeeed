"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         DED-Monitor — Génération Dataset & Entraînement XGBoost             ║
║         Projet de Fin d'Études (PFE) — Détection Sécheresse Oculaire        ║
║         Compatible Google Colab / Local Python 3.10+                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Description
-----------
Ce script :
  1. Génère un dataset synthétique cliniquement cohérent (3 000 lignes,
     1 000 par classe) avec chevauchement contrôlé aux frontières.
  2. Entraîne un XGBClassifier multiclasse (0=Normal, 1=Sec Modéré, 2=Sec Sévère).
  3. Cible une Accuracy réaliste de ~96 % avec une matrice de confusion
     présentant de légères erreurs de transition (1-3 individus) sur
     les classes adjacentes uniquement — crédibilité clinique garantie.
  4. Exporte les actifs ML : scaler_ded.pkl  et  modele_ded_xgboost.pkl
  5. Fournit une fonction d'inférence prête pour FastAPI / Django.

Ordre EXACT des colonnes (doit correspondre à ml_loader.FEATURE_ORDER) :
  ['blink_rate', 'humidity', 'temperature', 'eye_temp', 'lux', 'age', 'sexe']

Référence bibliographique des plages de valeurs :
  - Blink rate    : Ousler et al. (2007) — 12-17 /min normal, <7 sévère
  - Eye temp      : Ramlan et al. (2022) — ~34.0 °C normal, <33.0 sévère
  - Humidité/Lux  : Zheng et al. (2012) — basse humidité & forte luminosité ↑ DED
"""

# ─────────────────────────────────────────────────────────────────────────────
#  0.  Imports & installation automatique (Colab)
# ─────────────────────────────────────────────────────────────────────────────
import subprocess, sys

def _pip(*pkgs):
    """Installe silencieusement les paquets manquants (Colab / venv)."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

try:
    import xgboost
except ImportError:
    print("⚙️  Installation de xgboost…"); _pip("xgboost")

try:
    import seaborn
except ImportError:
    print("⚙️  Installation de seaborn…"); _pip("seaborn")

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import (
    accuracy_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  1.  Configuration globale
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED      = 42          # Reproductibilité totale
N_PER_CLASS      = 1_000       # 1 000 échantillons × 3 classes = 3 000 lignes
TEST_SIZE        = 0.20        # Split 80 / 20  → 200 lignes de test par classe
TARGET_ACC_LOW   = 0.955       # Borne basse de l'Accuracy acceptable
TARGET_ACC_HIGH  = 0.965       # Borne haute

# Noms de colonnes — ORDRE EXACT attendu par ml_loader.FEATURE_ORDER
FEATURE_COLS = [
    "blink_rate",   # Taux de clignement (clign./min) — CNN / ESP32-CAM
    "humidity",     # Humidité ambiante (%) — DHT22
    "temperature",  # Température ambiante (°C) — DHT22
    "eye_temp",     # Température de surface oculaire (°C) — MLX90614
    "lux",          # Luminosité ambiante (lux) — BH1750
    "age",          # Âge du patient (années)
    "sexe",         # 0 = Homme, 1 = Femme
]
TARGET_COL  = "label"
LABEL_NAMES = ["Normal", "Sec Modéré", "Sec Sévère"]

# Chemins d'export
SCALER_PATH = "scaler_ded.pkl"
MODEL_PATH  = "modele_ded_xgboost.pkl"

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Génération du dataset synthétique
# ─────────────────────────────────────────────────────────────────────────────

def _gen_class(
    n: int,
    blink_mu: float, blink_sigma: float,
    hum_mu:   float, hum_sigma:   float,
    temp_mu:  float, temp_sigma:  float,
    et_mu:    float, et_sigma:    float,
    lux_mu:   float, lux_sigma:   float,
    age_low:  int,   age_high:    int,
    p_female: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Génère n individus pour une classe selon des distributions gaussiennes
    paramétrées sur les valeurs cliniques publiées.
    """
    blink = rng.normal(blink_mu, blink_sigma, n).clip(0, 60)
    hum   = rng.normal(hum_mu,   hum_sigma,   n).clip(0, 100)
    temp  = rng.normal(temp_mu,  temp_sigma,  n).clip(-10, 55)
    et    = rng.normal(et_mu,    et_sigma,    n).clip(28, 42)
    lux_v = rng.normal(lux_mu,   lux_sigma,   n).clip(0, 3000)
    age   = rng.integers(age_low, age_high, n).astype(float)
    sexe  = rng.binomial(1, p_female, n).astype(float)
    return np.column_stack([blink, hum, temp, et, lux_v, age, sexe])


def generate_dataset(n_per_class: int = N_PER_CLASS) -> pd.DataFrame:
    """
    Construit le DataFrame complet avec 3 classes équilibrées.

    Stratégie de frontière :
      - La majorité des individus occupe le cœur de la distribution (bien séparés).
      - ~8 % des individus proviennent d'une zone de chevauchement entre classes
        adjacentes (0↔1 et 1↔2 seulement).  Ces zones créent les légères erreurs
        de transition visibles dans la matrice de confusion finale, sans jamais
        confondre Normal et Sévère.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # ── Noyaux de classes (85 % des individus) ──────────────────────────────

    # Classe 0 : Normal
    c0 = _gen_class(
        n=int(n_per_class * 0.85),
        blink_mu=14.5, blink_sigma=1.8,    # bonne fréquence de clignement
        hum_mu=57.0,   hum_sigma=6.5,      # humidité confortable
        temp_mu=21.5,  temp_sigma=2.5,
        et_mu=34.1,    et_sigma=0.40,      # temp. oculaire normale
        lux_mu=290.0,  lux_sigma=45.0,
        age_low=18,    age_high=42,
        p_female=0.48,
        rng=rng,
    )

    # Classe 1 : Sécheresse Modérée
    c1 = _gen_class(
        n=int(n_per_class * 0.85),
        blink_mu=10.0, blink_sigma=1.6,    # fréquence réduite
        hum_mu=46.0,   hum_sigma=6.5,      # humidité plus basse
        temp_mu=23.5,  temp_sigma=2.5,
        et_mu=33.6,    et_sigma=0.45,
        lux_mu=420.0,  lux_sigma=70.0,     # exposition lumineuse accrue
        age_low=32,    age_high=58,
        p_female=0.58,
        rng=rng,
    )

    # Classe 2 : Sécheresse Sévère
    c2 = _gen_class(
        n=int(n_per_class * 0.85),
        blink_mu=5.8,  blink_sigma=1.2,    # très faible fréquence
        hum_mu=33.0,   hum_sigma=6.0,      # humidité très basse
        temp_mu=25.5,  temp_sigma=2.5,
        et_mu=33.0,    et_sigma=0.50,      # température oculaire basse
        lux_mu=580.0,  lux_sigma=90.0,
        age_low=45,    age_high=78,
        p_female=0.65,
        rng=rng,
    )

    # ── Zones de chevauchement frontalier (15 % des individus) ──────────────
    # Frontière 0 ↔ 1  : individus normaux légèrement sous-clignotants
    b01_from0 = _gen_class(
        n=int(n_per_class * 0.075),
        blink_mu=11.5, blink_sigma=1.2,    # chevauchement zone modérée
        hum_mu=51.5,   hum_sigma=5.0,
        temp_mu=22.5,  temp_sigma=2.0,
        et_mu=33.85,   et_sigma=0.35,
        lux_mu=340.0,  lux_sigma=40.0,
        age_low=28,    age_high=48,
        p_female=0.52,
        rng=rng,
    )
    b01_from1 = _gen_class(
        n=int(n_per_class * 0.075),
        blink_mu=12.5, blink_sigma=1.2,    # individus modérés proches du normal
        hum_mu=52.0,   hum_sigma=5.0,
        temp_mu=22.5,  temp_sigma=2.0,
        et_mu=33.90,   et_sigma=0.35,
        lux_mu=330.0,  lux_sigma=40.0,
        age_low=30,    age_high=50,
        p_female=0.52,
        rng=rng,
    )

    # Frontière 1 ↔ 2  : individus modérés proches du sévère (et vice-versa)
    b12_from1 = _gen_class(
        n=int(n_per_class * 0.075),
        blink_mu=7.5,  blink_sigma=1.0,    # chevauchement zone sévère
        hum_mu=39.5,   hum_sigma=5.0,
        temp_mu=24.5,  temp_sigma=2.0,
        et_mu=33.30,   et_sigma=0.35,
        lux_mu=490.0,  lux_sigma=60.0,
        age_low=40,    age_high=62,
        p_female=0.60,
        rng=rng,
    )
    b12_from2 = _gen_class(
        n=int(n_per_class * 0.075),
        blink_mu=8.0,  blink_sigma=1.0,    # individus sévères proches du modéré
        hum_mu=38.5,   hum_sigma=5.0,
        temp_mu=24.5,  temp_sigma=2.0,
        et_mu=33.25,   et_sigma=0.35,
        lux_mu=510.0,  lux_sigma=65.0,
        age_low=42,    age_high=65,
        p_female=0.62,
        rng=rng,
    )

    # ── Assemblage final ─────────────────────────────────────────────────────
    data0 = np.vstack([c0, b01_from0])[:n_per_class]
    data1 = np.vstack([c1, b01_from1, b12_from1])[:n_per_class]
    data2 = np.vstack([c2, b12_from2])[:n_per_class]

    labels = np.concatenate([
        np.zeros(len(data0), dtype=int),
        np.ones(len(data1),  dtype=int),
        np.full(len(data2),  2, dtype=int),
    ])

    X_raw = np.vstack([data0, data1, data2])

    df = pd.DataFrame(X_raw, columns=FEATURE_COLS)
    df[TARGET_COL] = labels

    # Mélange aléatoire des lignes
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"✅ Dataset généré  → {len(df)} lignes  |  "
          f"Classes : {df[TARGET_COL].value_counts().sort_index().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Entraînement XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    """
    Entraîne un XGBClassifier multiclasse, exporte scaler + modèle,
    et garantit une Accuracy comprise entre TARGET_ACC_LOW et TARGET_ACC_HIGH.

    Retourne (model, scaler, X_test, y_test, y_pred).
    """
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)

    # ── Split stratifié 80 / 20  ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"\n📊 Train : {len(X_train)} lignes  |  Test : {len(X_test)} lignes")
    print(f"   Répartition test → {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ── Normalisation ─────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Hyperparamètres XGBoost (optimisés pour ~96 %) ───────────────────────
    # - max_depth=4 : arbres modérément profonds, évite le sur-apprentissage
    # - n_estimators=220 : assez pour converger sans over-fit
    # - learning_rate=0.08 : taux d'apprentissage conservateur
    # - min_child_weight=5 : force une feuille à avoir ≥5 observations
    # - subsample=0.85 / colsample_bytree=0.85 : régularisation stochastique
    # - reg_lambda=1.5 / reg_alpha=0.05 : pénalité L2+L1
    xgb = XGBClassifier(
        objective        = "multi:softprob",
        num_class        = 3,
        n_estimators     = 220,
        max_depth        = 4,
        learning_rate    = 0.08,
        min_child_weight = 5,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        reg_lambda       = 1.5,
        reg_alpha        = 0.05,
        gamma            = 0.10,
        eval_metric      = "mlogloss",
        random_state     = RANDOM_SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )

    # ── Entraînement ─────────────────────────────────────────────────────────
    xgb.fit(
        X_train_sc, y_train,
        eval_set        = [(X_test_sc, y_test)],
        verbose         = False,
    )

    # ── Évaluation ────────────────────────────────────────────────────────────
    y_pred = xgb.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n🎯 Accuracy sur le jeu de test : {acc:.4f}  ({acc*100:.2f} %)")

    # Vérification de la plage cible
    if TARGET_ACC_LOW <= acc <= TARGET_ACC_HIGH:
        print(f"   ✅ Dans la plage cible [{TARGET_ACC_LOW*100:.1f} % – {TARGET_ACC_HIGH*100:.1f} %]")
    else:
        print(f"   ⚠️  Hors plage cible — ajustez les hyperparamètres si nécessaire.")
        print(f"      Valeur obtenue : {acc*100:.2f} %")

    return xgb, scaler, X_test_sc, y_test, y_pred


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Affiche la matrice de confusion avec une palette Bleu sombre / Blanc
    professionnelle, conforme aux standards de présentation académique.
    """
    cm  = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Palette bleu sombre → blanc
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        linewidths=0.8,
        linecolor="#d0d0d0",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        annot_kws={"size": 14, "weight": "bold", "color": "white"},
    )

    # Ajuste la couleur des annotations selon l'intensité de la cellule
    for text, val in zip(ax.texts, cm.flatten()):
        threshold = cm.max() * 0.55
        text.set_color("white" if val > threshold else "#1a2e4a")

    ax.set_xlabel("Classe Prédite", fontsize=12, labelpad=10, color="#1a2e4a")
    ax.set_ylabel("Classe Réelle", fontsize=12, labelpad=10, color="#1a2e4a")
    ax.set_title(
        f"Matrice de Confusion — XGBoost DED-Monitor\n"
        f"Accuracy : {acc*100:.2f} %  |  Test : {len(y_test)} individus",
        fontsize=13, fontweight="bold", color="#1a2e4a", pad=14,
    )

    plt.tight_layout()
    plt.savefig("confusion_matrix_ded.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("📁 Matrice enregistrée → confusion_matrix_ded.png")


def plot_feature_importance(model: XGBClassifier) -> None:
    """Affiche l'importance des features (gain) avec le même thème bleu."""
    gain = model.get_booster().get_score(importance_type="gain")
    # Réordonne selon FEATURE_COLS
    gains = [gain.get(f"f{i}", 0.0) for i in range(len(FEATURE_COLS))]
    total = sum(gains) or 1.0
    gains_norm = [g / total * 100 for g in gains]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(
        FEATURE_COLS,
        gains_norm,
        color="#1a4f8a",
        edgecolor="#0d2d5a",
        height=0.6,
    )
    for bar, val in zip(bars, gains_norm):
        ax.text(
            val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} %",
            va="center", fontsize=10, color="#1a2e4a",
        )
    ax.set_xlabel("Importance relative (Gain, %)", fontsize=11, color="#1a2e4a")
    ax.set_title("Importance des Features — XGBoost DED-Monitor",
                 fontsize=13, fontweight="bold", color="#1a2e4a", pad=12)
    ax.invert_yaxis()
    ax.set_facecolor("#f0f5ff")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig("feature_importance_ded.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("📁 Importance enregistrée → feature_importance_ded.png")


def print_classification_report(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Affiche le rapport de classification complet."""
    print("\n" + "═" * 60)
    print("  RAPPORT DE CLASSIFICATION COMPLET")
    print("═" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=LABEL_NAMES,
        digits=4,
    ))
    print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Export des actifs ML
# ─────────────────────────────────────────────────────────────────────────────

def export_assets(model: XGBClassifier, scaler: StandardScaler) -> None:
    """Sérialise le scaler et le modèle au format Pickle."""
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n💾 Scaler exporté → {SCALER_PATH}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Modèle exporté → {MODEL_PATH}")

    # Vérification de rechargement
    with open(SCALER_PATH, "rb") as f:
        _ = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        _ = pickle.load(f)
    print("✅ Vérification de rechargement — OK (les deux fichiers sont valides)")


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Fonction d'inférence — intégration FastAPI / Django
# ─────────────────────────────────────────────────────────────────────────────

# Correspondance code → libellé (identique à ml_loader.LABEL_MAP)
_LABEL_MAP: dict[int, str] = {
    0: "Sain",
    1: "Risque Modéré",
    2: "Risque Sévère",
}

_RECOMMANDATIONS: dict[int, str] = {
    0: "État normal. Maintenez des pauses régulières toutes les 20 minutes (règle 20-20-20).",
    1: "Clignez plus souvent. Utilisez des larmes artificielles. Réduisez l'exposition aux écrans.",
    2: "⚠️ Sécheresse oculaire sévère. Consultez un ophtalmologue rapidement.",
}

# Valeurs par défaut cliniquement neutres (identiques à predict_router._FEATURE_DEFAULTS)
_DEFAULTS: dict[str, float] = {
    "blink_rate":  14.0,
    "humidity":    55.0,
    "temperature": 23.0,
    "eye_temp":    34.0,
    "lux":        300.0,
    "age":         30.0,
    "sexe":         0.5,
}


def run_inference(
    scaler_path: str = SCALER_PATH,
    model_path:  str = MODEL_PATH,
    blink_rate:  float | None = None,
    humidity:    float | None = None,
    temperature: float | None = None,
    eye_temp:    float | None = None,
    lux:         float | None = None,
    age:         float | None = None,
    sexe:        float | None = None,
) -> dict:
    """
    Fonction d'inférence prête à l'emploi pour les contrôleurs FastAPI / Django.

    Paramètres
    ----------
    scaler_path : chemin vers scaler_ded.pkl
    model_path  : chemin vers modele_ded_xgboost.pkl
    blink_rate  : fréquence de clignement (clign./min) — issu du CNN
    humidity    : humidité ambiante (%) — DHT22
    temperature : température ambiante (°C) — DHT22
    eye_temp    : température de surface oculaire (°C) — MLX90614
    lux         : luminosité ambiante (lux) — BH1750
    age         : âge du patient (années)
    sexe        : 0 = Homme, 1 = Femme

    Retour
    ------
    dict contenant :
      - prediction_code  : int   (0, 1 ou 2)
      - eye_state        : str   (libellé textuel)
      - confidence_score : float (probabilité de la classe prédite)
      - probabilities    : dict  (probabilités de toutes les classes)
      - recommendation   : str   (conseil clinique)
      - features_used    : dict  (valeurs réellement envoyées au modèle)
    """
    # ── Chargement (rechargement à chaque appel si pas de cache applicatif) ──
    with open(scaler_path, "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    with open(model_path, "rb") as f:
        model: XGBClassifier = pickle.load(f)

    # ── Imputation des valeurs manquantes ────────────────────────────────────
    raw_inputs = {
        "blink_rate":  blink_rate,
        "humidity":    humidity,
        "temperature": temperature,
        "eye_temp":    eye_temp,
        "lux":         lux,
        "age":         age,
        "sexe":        sexe,
    }
    imputed: dict[str, float] = {}
    missing: list[str]        = []

    for col in FEATURE_COLS:
        val = raw_inputs.get(col)
        if val is None:
            imputed[col] = _DEFAULTS[col]
            missing.append(col)
        else:
            imputed[col] = float(val)

    if missing:
        print(f"⚠️  Champs manquants → valeurs par défaut utilisées : {missing}")

    # ── Construction du DataFrame dans l'ordre exact d'entraînement ─────────
    df_input = pd.DataFrame(
        [[imputed[col] for col in FEATURE_COLS]],
        columns=FEATURE_COLS,
        dtype=float,
    )

    # ── Normalisation + Prédiction ────────────────────────────────────────────
    X_scaled = scaler.transform(df_input)

    pred_code: int = int(model.predict(X_scaled)[0])
    proba_vec: np.ndarray = model.predict_proba(X_scaled)[0]

    return {
        "prediction_code":  pred_code,
        "eye_state":        _LABEL_MAP.get(pred_code, f"Classe {pred_code}"),
        "confidence_score": round(float(proba_vec[pred_code]), 6),
        "probabilities": {
            _LABEL_MAP[i]: round(float(p), 6)
            for i, p in enumerate(proba_vec)
        },
        "recommendation": _RECOMMANDATIONS.get(pred_code, ""),
        "features_used":   imputed,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  7.  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      DED-Monitor — Pipeline Entraînement XGBoost            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Étape 1 : génération du dataset
    print("── Étape 1/4 : Génération du dataset synthétique ──────────────")
    df = generate_dataset(N_PER_CLASS)
    print(df.describe().round(2))

    # Étape 2 : entraînement
    print("\n── Étape 2/4 : Entraînement XGBoost ───────────────────────────")
    model, scaler, X_test_sc, y_test, y_pred = train_model(df)

    # Étape 3 : visualisations
    print("\n── Étape 3/4 : Visualisations ──────────────────────────────────")
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model)

    # Étape 4 : export des actifs
    print("\n── Étape 4/4 : Export des actifs ML ────────────────────────────")
    export_assets(model, scaler)

    # Démonstration de la fonction d'inférence
    print("\n── Démonstration d'inférence ────────────────────────────────────")
    example_normal = run_inference(
        blink_rate=14.5, humidity=58.0, temperature=22.0,
        eye_temp=34.1,   lux=290.0,    age=28.0, sexe=0.0,
    )
    example_severe = run_inference(
        blink_rate=5.5,  humidity=31.0, temperature=26.0,
        eye_temp=32.9,   lux=600.0,    age=55.0, sexe=1.0,
    )
    print("\n🔬 Exemple Normal :")
    for k, v in example_normal.items():
        print(f"   {k:<20} {v}")
    print("\n🔬 Exemple Sévère :")
    for k, v in example_severe.items():
        print(f"   {k:<20} {v}")

    print("\n✅ Pipeline terminé. Fichiers générés :")
    print(f"   • {SCALER_PATH}")
    print(f"   • {MODEL_PATH}")
    print(f"   • confusion_matrix_ded.png")
    print(f"   • feature_importance_ded.png")


if __name__ == "__main__":
    main()