"""
inference_engine.py
===================
Motor de inferencia LOF con calibracion correcta del Indice de Salud.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from core.feature_extractor import extract_features
from core.model_trainer import load_artifacts, DEFAULT_MODEL_DIR

logger = logging.getLogger(__name__)

STATUS_NORMAL = "NORMAL"
STATUS_WARN   = "ADVERTENCIA"
STATUS_ALERT  = "ALERTA"

LOF_WEIGHT = 1.0
IF_WEIGHT  = 0.0


@dataclass
class InferenceResult:
    status: str = STATUS_NORMAL
    health_index: float = 100.0
    anomaly_score: float = 0.0
    feature_vector: np.ndarray = field(default_factory=lambda: np.zeros(80))
    machine_id: str = "unknown"
    file_name: str = ""
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_anomaly(self) -> bool:
        return self.status != STATUS_NORMAL

    @property
    def status_emoji(self) -> str:
        return {"NORMAL": "✅", "ADVERTENCIA": "⚠️", "ALERTA": "🚨"}.get(self.status, "❓")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "health_index": round(self.health_index, 2),
            "anomaly_score": round(float(self.anomaly_score), 6),
            "machine_id": self.machine_id,
            "file_name": self.file_name,
            "is_anomaly": self.is_anomaly,
            "error": self.error,
            "metadata": self.metadata,
        }


class InferenceEngine:

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self._lof    = None
        self._scaler = None
        self._meta: dict = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._lof, _, self._scaler, self._meta = load_artifacts(self.model_dir)
        logger.info("Modelo cargado — AUC: %.4f | Recall: %.4f",
                    self._meta.get("auc_validation", 0.0),
                    self._meta.get("recall_validation", 0.0))
        self._loaded = True

    def _score_to_health(self, raw_score: float) -> float:
        """
        Mapea el score LOF crudo a Indice de Salud [0-100%].

        Anclas calibradas con percentiles del set de entrenamiento:
          score <= score_min_normal  → salud = 100% (muy normal)
          score >= threshold_alert   → salud = 0%   (muy anomalo)
        """
        s_min  = self._meta.get("score_min_normal", -0.5)  # muy normal
        s_alert = self._meta.get("threshold_alert", -0.1)  # frontera alerta

        # Invertir: score bajo = normal = salud alta
        normalized = (raw_score - s_min) / (s_alert - s_min + 1e-9)
        health = (1.0 - normalized) * 100.0
        return float(np.clip(health, 0.0, 100.0))

    def _determine_status(self, raw_score: float) -> str:
        """
        Tres zonas calibradas con percentiles reales:
          NORMAL      : score < threshold_warn   (90% percentil normales)
          ADVERTENCIA : score < threshold_alert  (umbral optimizado Recall)
          ALERTA      : score >= threshold_alert
        """
        thresh_warn  = self._meta.get("threshold_warn",  -0.25)
        thresh_alert = self._meta.get("threshold_alert", -0.15)
        if raw_score < thresh_warn:
            return STATUS_NORMAL
        elif raw_score < thresh_alert:
            return STATUS_WARN
        else:
            return STATUS_ALERT

    def predict(self, audio_path: str | Path,
                machine_id: str = "unknown") -> InferenceResult:

        audio_path = Path(audio_path)
        result = InferenceResult(machine_id=machine_id,
                                 file_name=audio_path.name,
                                 metadata=self._meta)
        try:
            self._load()
        except FileNotFoundError as exc:
            result.error = str(exc)
            result.status = "ERROR"
            return result

        try:
            features = extract_features(audio_path)
        except FileNotFoundError as exc:
            result.error = f"Archivo no encontrado: {exc}"
            result.status = "ERROR"
            return result
        except ValueError as exc:
            result.error = f"Audio corrupto: {exc}"
            result.status = "ERROR"
            return result
        except Exception as exc:
            result.error = f"Error extrayendo features: {exc}"
            result.status = "ERROR"
            return result

        result.feature_vector = features

        try:
            X        = features.reshape(1, -1)
            X_scaled = self._scaler.transform(X)
            raw_score = float(-self._lof.decision_function(X_scaled)[0])
        except Exception as exc:
            result.error = f"Error en inferencia: {exc}"
            result.status = "ERROR"
            return result

        result.anomaly_score = raw_score
        result.health_index  = self._score_to_health(raw_score)
        result.status        = self._determine_status(raw_score)
        return result


_global_engine: Optional[InferenceEngine] = None


def predict(audio_path: str, machine_id: str = "unknown",
            model_dir: str = str(DEFAULT_MODEL_DIR)) -> dict:
    global _global_engine
    if _global_engine is None:
        _global_engine = InferenceEngine(model_dir=model_dir)
    return _global_engine.predict(audio_path, machine_id=machine_id).to_dict()