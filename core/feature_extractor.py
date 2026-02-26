"""
feature_extractor.py
====================
Módulo de extracción de características acústicas para el sistema
de detección de anomalías en bombas industriales (MIMII Dataset - 6dB SNR).

Vector de salida: 80 dimensiones
  - Temporales   (15): RMS, Crest Factor, Kurtosis, Skewness, ZCR, etc.
  - PSD / Welch  (25): Densidad espectral de potencia por bandas
  - Onsets       (10): Descriptores de detección de impactos
  - MFCCs        (30): 13 coef + deltas + dobles deltas

Frecuencia de muestreo objetivo: 8 000 Hz
  → Nyquist limit: 4 000 Hz (rango mecánico relevante en bombas centrífugas)

Autor  : Proyecto de Egreso – Ingeniería Mecatrónica, UNI
Dataset: MIMII Dataset (Pump, +6 dB SNR)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes del pipeline
# ---------------------------------------------------------------------------
SR_TARGET: int = 8_000          # Hz  — Nyquist = 4 000 Hz
N_FFT: int = 1_024
HOP_LENGTH: int = 256
MAX_DURATION: float = 10.0      # segundos máximos de audio
N_MFCC: int = 13
N_FEATURES: int = 80


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _safe_scalar(value) -> float:
    """Convierte cualquier valor a float escalar de forma segura."""
    try:
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            return float(np.nanmean(arr)) if arr.size > 0 else 0.0
        return float(value)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def extract_features(
    audio_path: str | Path,
    sr: int = SR_TARGET,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    max_duration: float = MAX_DURATION,
) -> np.ndarray:
    """
    Extrae un vector de características de 80 dimensiones a partir de un archivo WAV.

    Parameters
    ----------
    audio_path  : Ruta al archivo de audio (.wav)
    sr          : Frecuencia de muestreo objetivo (Hz)
    n_fft       : Tamaño de la ventana FFT
    hop_length  : Desplazamiento entre ventanas
    max_duration: Duración máxima de audio a procesar (segundos)

    Returns
    -------
    np.ndarray de forma (80,), dtype float32

    Raises
    ------
    FileNotFoundError : Si el archivo no existe
    ValueError        : Si el audio está vacío o corrupto
    RuntimeError      : Si la extracción falla por otra razón
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {audio_path}")

    # ── Carga ────────────────────────────────────────────────────────────────
    try:
        y, _ = librosa.load(
            str(audio_path),
            sr=sr,
            mono=True,
            duration=max_duration,
        )
    except Exception as exc:
        raise ValueError(f"Error al cargar el audio '{audio_path.name}': {exc}") from exc

    if y is None or len(y) == 0:
        raise ValueError(f"El archivo '{audio_path.name}' está vacío o corrupto.")

    # Normalización de amplitud
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    features: list[float] = []

    # ── BLOQUE 1: Descriptores Temporales (15 features) ─────────────────────
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak_val = float(np.max(np.abs(y)))
    crest_factor = peak_val / (rms + 1e-9)
    kurt = float(kurtosis(y, fisher=True))
    skewness = float(skew(y))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, hop_length=hop_length)))

    # Shape factor, Impulse factor, Margin factor
    mean_abs = float(np.mean(np.abs(y)))
    shape_factor = rms / (mean_abs + 1e-9)
    impulse_factor = peak_val / (mean_abs + 1e-9)
    margin_factor = peak_val / ((np.mean(np.sqrt(np.abs(y))) + 1e-9) ** 2)

    # Estadísticas de percentiles
    p25, p75 = float(np.percentile(np.abs(y), 25)), float(np.percentile(np.abs(y), 75))
    iqr = p75 - p25
    energy = float(np.sum(y ** 2))
    variance = float(np.var(y))
    std_dev = float(np.std(y))
    mean_env = float(np.mean(np.abs(librosa.effects.preemphasis(y))))

    features.extend([
        rms, peak_val, crest_factor, kurt, skewness,
        zcr, shape_factor, impulse_factor, margin_factor, iqr,
        energy, variance, std_dev, mean_env, p75,
    ])  # 15

    # ── BLOQUE 2: PSD / Welch – Bandas de frecuencia (25 features) ──────────
    nperseg = min(256, len(y))
    freqs, psd = welch(y, fs=sr, nperseg=nperseg)

    # Bandas mecánicas relevantes [Hz] para bombas centrífugas
    bands = [
        (0, 100), (100, 300), (300, 600), (600, 1000),
        (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000),
        (3000, 3500), (3500, 4000),
    ]
    band_powers: list[float] = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_powers.append(float(np.sum(psd[mask])) if mask.any() else 0.0)

    total_power = sum(band_powers) + 1e-12
    band_ratios = [p / total_power for p in band_powers]

    # Centroide espectral, ancho de banda, flujo
    spec_centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
    freq_sq = freqs ** 2
    spec_bandwidth = float(np.sqrt(np.sum(freq_sq * psd) / (np.sum(psd) + 1e-12)))
    spec_flatness = float(np.exp(np.mean(np.log(psd + 1e-12))) / (np.mean(psd) + 1e-12))
    peak_freq_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_freq_idx])
    spectral_entropy = float(-np.sum((psd / total_power) * np.log(psd / total_power + 1e-12)))

    features.extend(band_powers)          # 10
    features.extend(band_ratios[:5])      # 5
    features.extend([
        spec_centroid, spec_bandwidth,
        spec_flatness, peak_freq, spectral_entropy,
        float(np.max(psd)), float(np.min(psd)),
        float(np.median(psd)), float(np.std(psd)),
        float(np.sum(psd[(freqs >= 2000) & (freqs < 4000)])),
    ])  # 10 → total bloque = 25

    # ── BLOQUE 3: Detección de Impactos / Onsets (10 features) ──────────────
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length, n_fft=n_fft
    )
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length
    )

    onset_rate = len(onsets) / (len(y) / sr + 1e-9)
    onset_env_mean = float(np.mean(onset_env))
    onset_env_std = float(np.std(onset_env))
    onset_env_max = float(np.max(onset_env))
    onset_env_kurtosis = float(kurtosis(onset_env, fisher=True))

    # Intervalos entre onsets
    if len(onsets) > 1:
        intervals = np.diff(onsets).astype(float)
        ioi_mean = float(np.mean(intervals))
        ioi_std = float(np.std(intervals))
        ioi_min = float(np.min(intervals))
    else:
        ioi_mean, ioi_std, ioi_min = 0.0, 0.0, 0.0

    onset_env_entropy = float(-np.sum(
        (onset_env / (np.sum(onset_env) + 1e-12)) *
        np.log(onset_env / (np.sum(onset_env) + 1e-12) + 1e-12)
    ))
    onset_regularity = 1.0 / (ioi_std + 1.0)

    features.extend([
        onset_rate, onset_env_mean, onset_env_std,
        onset_env_max, onset_env_kurtosis,
        ioi_mean, ioi_std, ioi_min,
        onset_env_entropy, onset_regularity,
    ])  # 10

    # ── BLOQUE 4: MFCCs + Deltas (30 features) ───────────────────────────────
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC,
        n_fft=n_fft, hop_length=hop_length,
        fmax=sr // 2,
    )
    delta1 = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    mfcc_means = np.mean(mfccs, axis=1)    # 13
    delta1_means = np.mean(delta1, axis=1)  # 13 → usamos solo 4 pares
    delta2_means = np.mean(delta2, axis=1)  # 13 → usamos solo 3

    # Seleccionamos 13 + 9 + 8 = 30 para cerrar en 80
    features.extend(mfcc_means.tolist())          # 13
    features.extend(delta1_means[:9].tolist())    # 9
    features.extend(delta2_means[:8].tolist())    # 8 → total = 30

    # ── Validación y ajuste de dimensión ────────────────────────────────────
    feature_vector = np.array([_safe_scalar(f) for f in features], dtype=np.float32)

    if len(feature_vector) < N_FEATURES:
        padding = np.zeros(N_FEATURES - len(feature_vector), dtype=np.float32)
        feature_vector = np.concatenate([feature_vector, padding])
    elif len(feature_vector) > N_FEATURES:
        feature_vector = feature_vector[:N_FEATURES]

    if not np.all(np.isfinite(feature_vector)):
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_vector  # shape: (80,)
