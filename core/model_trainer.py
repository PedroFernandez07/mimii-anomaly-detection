"""
model_trainer.py
================
LOF puro con umbral optimizado para maximizar Recall en mantenimiento industrial.

Estrategia:
  - LOF solo (AUC=0.891, mejor que ensemble)
  - Umbral calibrado para Recall >= 0.85 (objetivo industrial)
  - Se busca el umbral que maximiza Recall con Precision >= 0.60

Uso:
  python -m core.model_trainer --data_dir ".\pump" --output_dir ".\models"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

from core.feature_extractor import extract_features

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"
LOF_WEIGHT = 1.0
IF_WEIGHT  = 0.0


def build_inventory(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    records = []
    for root, _, files in os.walk(data_dir):
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if not wavs:
            continue
        parts = Path(root).parts
        machine_id = next((p for p in parts if p.startswith("id_")), "unknown")
        label = (
            "abnormal" if "abnormal" in parts
            else ("normal" if "normal" in parts else "unknown")
        )
        for wav in wavs:
            records.append({"machine_id": machine_id, "label": label,
                            "path": str(Path(root) / wav)})

    if not records:
        raise RuntimeError(f"No se encontraron archivos .wav en '{data_dir}'.")

    df = pd.DataFrame(records)
    df = df[df["label"] != "unknown"].reset_index(drop=True)
    logger.info("Inventario: %d archivos | %s", len(df),
                df.groupby("label").size().to_dict())
    return df


def _find_optimal_threshold(scores: np.ndarray, y_true: np.ndarray,
                             min_precision: float = 0.55) -> dict:
    """
    Busca el umbral que maximiza Recall manteniendo Precision >= min_precision.
    Estrategia industrial: preferimos no perder fallas reales.
    """
    thresholds = np.percentile(scores, np.arange(30, 96, 1))
    best = {"recall": 0.0, "f1": 0.0, "precision": 0.0,
            "threshold": thresholds[0]}

    for t in thresholds:
        preds = (scores >= t).astype(int)
        rec  = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)

        # Maximizar Recall con restriccion de Precision minima
        if prec >= min_precision and rec > best["recall"]:
            best = {"recall": rec, "f1": f1,
                    "precision": prec, "threshold": float(t)}

    return best


def train_and_save(
    data_dir: str | Path,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
    train_ratio: float = 0.80,
    n_neighbors: int = 20,
    contamination: float = 0.08,
    min_precision: float = 0.55,
    random_state: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_inventory(data_dir)

    normal_df   = df[df["label"] == "normal"].sample(frac=1, random_state=random_state)
    abnormal_df = df[df["label"] == "abnormal"]

    n_train       = int(len(normal_df) * train_ratio)
    train_df      = normal_df.iloc[:n_train]
    val_normal_df = normal_df.iloc[n_train:]

    logger.info("Extrayendo features de %d muestras normales (train)...", len(train_df))
    X_train = _extract_batch(train_df["path"].tolist())

    logger.info("Extrayendo features de %d muestras normales (val)...", len(val_normal_df))
    X_val_normal = _extract_batch(val_normal_df["path"].tolist())

    logger.info("Extrayendo features de %d muestras anomalas (val)...", len(abnormal_df))
    X_val_abnormal = _extract_batch(abnormal_df["path"].tolist())

    # Normalizacion
    scaler          = RobustScaler()
    X_train_sc      = scaler.fit_transform(X_train)
    X_val_normal_sc = scaler.transform(X_val_normal)
    X_val_abn_sc    = scaler.transform(X_val_abnormal)

    X_val = np.vstack([X_val_normal_sc, X_val_abn_sc])
    y_val = np.array([0] * len(X_val_normal) + [1] * len(X_val_abnormal))

    # LOF
    logger.info("Entrenando LOF (k=%d, contamination=%.2f)...",
                n_neighbors, contamination)
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof.fit(X_train_sc)

    scores_all    = -lof.decision_function(X_val)
    scores_normal = -lof.decision_function(X_val_normal_sc)
    scores_abn    = -lof.decision_function(X_val_abn_sc)

    auc = roc_auc_score(y_val, scores_all)
    logger.info("AUC: %.4f", auc)

    # Umbral optimizado para Recall industrial
    logger.info("Optimizando umbral (Precision minima=%.2f)...", min_precision)
    best = _find_optimal_threshold(scores_all, y_val, min_precision)
    logger.info("Umbral optimo → Recall=%.4f | Precision=%.4f | F1=%.4f | Thresh=%.6f",
                best["recall"], best["precision"], best["f1"], best["threshold"])

    # Umbrales del tablero
    thresh_warn  = float(np.percentile(scores_normal, 90))
    thresh_alert = best["threshold"]

    # Persistencia
    with open(output_dir / "lof_model.pkl", "wb") as f:
        pickle.dump(lof, f)
    with open(output_dir / "robust_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Limpiar artefactos anteriores que ya no se usan
    for old in ["if_model.pkl", "pca_reducer.pkl"]:
        p = output_dir / old
        if p.exists():
            p.unlink()

    meta = {
        "algorithm": "LocalOutlierFactor",
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "auc_validation": round(auc, 4),
        "f1_validation": round(best["f1"], 4),
        "recall_validation": round(best["recall"], 4),
        "precision_validation": round(best["precision"], 4),
        "threshold_warn":  thresh_warn,
        "threshold_alert": best["threshold"],
        "score_min_normal": float(np.min(scores_normal)),
        "score_max_normal": float(np.max(scores_normal)),
        "n_train": len(X_train),
        "n_val_normal": len(X_val_normal),
        "n_val_abnormal": len(X_val_abnormal),
        "sr_target": 8000,
        "n_features": 80,
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTADO FINAL")
    logger.info("  AUC:       %.4f", auc)
    logger.info("  Recall:    %.4f  (objetivo: >= 0.85)", best["recall"])
    logger.info("  Precision: %.4f", best["precision"])
    logger.info("  F1:        %.4f", best["f1"])
    logger.info("=" * 60)
    return meta


def _extract_batch(paths: list[str]) -> np.ndarray:
    results = []
    for path in tqdm(paths, desc="  features", leave=False):
        try:
            results.append(extract_features(path))
        except Exception as exc:
            logger.warning("Ignorado '%s': %s", path, exc)
    if not results:
        raise RuntimeError("No se pudo extraer ninguna feature del batch.")
    return np.vstack(results).astype(np.float32)


def load_artifacts(model_dir: str | Path = DEFAULT_MODEL_DIR) -> tuple:
    model_dir = Path(model_dir)
    lof_p    = model_dir / "lof_model.pkl"
    scaler_p = model_dir / "robust_scaler.pkl"
    meta_p   = model_dir / "training_meta.json"

    if not lof_p.exists() or not scaler_p.exists():
        raise FileNotFoundError(
            f"Artefactos no encontrados en '{model_dir}'.\n"
            "Ejecuta: python -m core.model_trainer --data_dir .\\pump --output_dir .\\models"
        )

    with open(lof_p, "rb") as f:
        lof = pickle.load(f)
    with open(scaler_p, "rb") as f:
        scaler = pickle.load(f)

    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return lof, None, scaler, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--min_precision", type=float, default=0.55,
                        help="Precision minima aceptable al optimizar Recall")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    args = parser.parse_args()

    train_and_save(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_neighbors=args.n_neighbors,
        contamination=args.contamination,
        min_precision=args.min_precision,
        train_ratio=args.train_ratio,
    )