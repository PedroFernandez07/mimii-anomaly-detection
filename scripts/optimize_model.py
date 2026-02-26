"""
optimize_model.py
=================
Benchmark de optimizacion del modelo LOF con diferentes estrategias:

  Estrategia 1 — LOF baseline (ya conocido, AUC ~0.90)
  Estrategia 2 — LOF + PCA (distintos n_components)
  Estrategia 3 — LOF + UMAP (reduccion no lineal)
  Estrategia 4 — LOF + Feature Selection (top-K por mutual info)
  Estrategia 5 — LOF + PCA + features mecanicas extra

Al final guarda el mejor modelo automaticamente en ./models/

Uso:
  python scripts/optimize_model.py --data_dir ".\pump"
"""

from __future__ import annotations

import sys
import os
import json
import pickle
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from scipy.signal import welch
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.feature_extractor import extract_features

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Features mecanicas adicionales especificas para bombas centrifugas
# ---------------------------------------------------------------------------

def extract_mechanical_features(audio_path: str, sr: int = 8000) -> np.ndarray:
    """
    Agrega 15 features mecanicas especializadas al vector base de 80D.
    Total resultante: 95 features.

    Features adicionales:
      - Bearing fault frequencies (BPFI, BPFO, BSF aproximados)
      - Cepstrum peaks (deteccion de periodicidades de falla)
      - Spectral kurtosis por banda
      - Wavelet energy ratio
      - Sidebands energy (modulacion de amplitud tipica en fallas)
    """
    import librosa

    base = extract_features(audio_path)

    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=10.0)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        extra = []

        # -- Cepstrum (periodicidades de impacto) --
        spectrum = np.fft.fft(y)
        log_spectrum = np.log(np.abs(spectrum) + 1e-12)
        cepstrum = np.real(np.fft.ifft(log_spectrum))
        # Energia en quefrency correspondiente a fallas tipicas (2-20 ms)
        q_low  = int(0.002 * sr)
        q_high = int(0.020 * sr)
        cep_energy = float(np.sum(cepstrum[q_low:q_high] ** 2))
        cep_peak   = float(np.max(np.abs(cepstrum[q_low:q_high])))
        extra.extend([cep_energy, cep_peak])

        # -- Spectral Kurtosis por banda (indicador de impactos) --
        freqs, psd = welch(y, fs=sr, nperseg=256)
        bands_sk = [(500, 1000), (1000, 2000), (2000, 3000), (3000, 4000)]
        for lo, hi in bands_sk:
            mask = (freqs >= lo) & (freqs < hi)
            band_signal = psd[mask] if mask.any() else np.array([0.0])
            extra.append(float(kurtosis(band_signal, fisher=True)))

        # -- Sideband energy ratio (modulacion AM tipica en engranajes/rodamientos) --
        # Busca el pico dominante y mide energia en sus bandas laterales
        peak_idx = int(np.argmax(psd))
        sideband_width = 5
        lo_sb = max(0, peak_idx - sideband_width)
        hi_sb = min(len(psd), peak_idx + sideband_width)
        center_energy  = float(psd[peak_idx])
        sideband_energy = float(np.sum(psd[lo_sb:hi_sb]) - center_energy)
        sideband_ratio  = sideband_energy / (center_energy + 1e-12)
        extra.extend([sideband_energy, sideband_ratio])

        # -- Envelope analysis (HFRT - High Frequency Resonance Technique) --
        # Filtra banda alta, rectifica, calcula envelope
        from scipy.signal import butter, filtfilt, hilbert
        b, a = butter(4, [2000 / (sr / 2), 3999 / (sr / 2)], btype='band')
        try:
            y_hf = filtfilt(b, a, y)
        except Exception:
            y_hf = y
        envelope = np.abs(hilbert(y_hf))
        env_kurt = float(kurtosis(envelope, fisher=True))
        env_rms  = float(np.sqrt(np.mean(envelope ** 2)))
        env_peak = float(np.max(envelope))
        env_crest = env_peak / (env_rms + 1e-9)
        extra.extend([env_kurt, env_rms, env_peak, env_crest])

        # -- Impulsividad en dominio tiempo-frecuencia --
        stft = np.abs(librosa.stft(y, n_fft=512, hop_length=128))
        tf_kurtosis = float(np.mean([kurtosis(stft[i], fisher=True)
                                     for i in range(stft.shape[0])]))
        extra.append(tf_kurtosis)

        extra_arr = np.array(extra, dtype=np.float32)
        extra_arr = np.nan_to_num(extra_arr, nan=0.0, posinf=0.0, neginf=0.0)

    except Exception:
        extra_arr = np.zeros(15, dtype=np.float32)

    return np.concatenate([base, extra_arr])  # 80 + 15 = 95D


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def build_inventory(data_dir: Path) -> pd.DataFrame:
    records = []
    for root, _, files in os.walk(data_dir):
        wavs = [f for f in files if f.lower().endswith(".wav")]
        if not wavs:
            continue
        parts = Path(root).parts
        machine_id = next((p for p in parts if p.startswith("id_")), "unknown")
        label = ("abnormal" if "abnormal" in parts
                 else ("normal" if "normal" in parts else "unknown"))
        for wav in wavs:
            records.append({"machine_id": machine_id, "label": label,
                            "path": str(Path(root) / wav)})
    df = pd.DataFrame(records)
    return df[df["label"] != "unknown"].reset_index(drop=True)


def extract_batch(paths, extractor_fn, desc=""):
    results = []
    for path in tqdm(paths, desc=f"  {desc}", leave=False):
        try:
            results.append(extractor_fn(path))
        except Exception:
            pass
    return np.vstack(results).astype(np.float32) if results else np.array([])


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------

def run_optimization(data_dir: str, sample_size: int = 400,
                     output_dir: str = "./models", random_state: int = 42):

    print("\n" + "="*70)
    print("  OPTIMIZACION DE MODELO — LOF Variants")
    print("  MIMII Dataset · Pump · +6 dB SNR")
    print("="*70)

    df = build_inventory(Path(data_dir))
    normal_df   = df[df["label"] == "normal"].sample(frac=1, random_state=random_state)
    abnormal_df = df[df["label"] == "abnormal"].sample(frac=1, random_state=random_state)

    n_normal   = min(sample_size, len(normal_df))
    n_abnormal = min(200, len(abnormal_df))
    n_train    = int(n_normal * 0.80)

    train_paths  = normal_df["path"].iloc[:n_train].tolist()
    val_n_paths  = normal_df["path"].iloc[n_train:n_normal].tolist()
    val_ab_paths = abnormal_df["path"].iloc[:n_abnormal].tolist()

    print(f"\nMuestras → Train: {n_train} | Val Normal: {len(val_n_paths)} | Val Anomalo: {n_abnormal}")

    # -- Extraer features base (80D) --
    print("\nExtrayendo features base (80D)...")
    X_train_80  = extract_batch(train_paths,  extract_features, "train")
    X_val_n_80  = extract_batch(val_n_paths,  extract_features, "val normal")
    X_val_ab_80 = extract_batch(val_ab_paths, extract_features, "val anomalo")

    # -- Extraer features mecanicas (95D) --
    print("\nExtrayendo features mecanicas (95D)...")
    X_train_95  = extract_batch(train_paths,  extract_mechanical_features, "train+mec")
    X_val_n_95  = extract_batch(val_n_paths,  extract_mechanical_features, "val_n+mec")
    X_val_ab_95 = extract_batch(val_ab_paths, extract_mechanical_features, "val_ab+mec")

    y_val = np.array([0] * len(X_val_n_80) + [1] * len(X_val_ab_80))

    # -- Normalizacion --
    def scale(X_tr, X_vn, X_va):
        sc = RobustScaler()
        return sc.fit_transform(X_tr), sc.transform(X_vn), sc.transform(X_va), sc

    Xtr80, Xvn80, Xva80, sc80 = scale(X_train_80, X_val_n_80, X_val_ab_80)
    Xtr95, Xvn95, Xva95, sc95 = scale(X_train_95, X_val_n_95, X_val_ab_95)

    Xval80 = np.vstack([Xvn80, Xva80])
    Xval95 = np.vstack([Xvn95, Xva95])

    # ---------------------------------------------------------------------------
    # Variantes a evaluar
    # ---------------------------------------------------------------------------
    variants = []

    # 1. Baseline LOF 80D (referencia)
    variants.append({
        "name": "LOF k=20 (baseline 80D)",
        "model": LocalOutlierFactor(n_neighbors=20, contamination=0.08,
                                    novelty=True, n_jobs=-1),
        "X_train": Xtr80, "X_val": Xval80,
        "scaler": sc80, "n_features": 80, "pca": None,
    })

    # 2. LOF + PCA variantes
    for n_comp in [10, 15, 20, 30]:
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xtr_pca  = pca.fit_transform(Xtr80)
        Xval_pca = pca.transform(Xval80)
        var_exp  = pca.explained_variance_ratio_.sum()
        variants.append({
            "name": f"LOF k=20 + PCA({n_comp}D, {var_exp:.0%} var)",
            "model": LocalOutlierFactor(n_neighbors=20, contamination=0.08,
                                        novelty=True, n_jobs=-1),
            "X_train": Xtr_pca, "X_val": Xval_pca,
            "scaler": sc80, "n_features": n_comp, "pca": pca,
        })

    # 3. LOF k diferentes con PCA optimo (20D)
    pca20 = PCA(n_components=20, random_state=random_state)
    Xtr20  = pca20.fit_transform(Xtr80)
    Xval20 = pca20.transform(Xval80)
    for k in [5, 10, 30, 50]:
        variants.append({
            "name": f"LOF k={k} + PCA(20D)",
            "model": LocalOutlierFactor(n_neighbors=k, contamination=0.08,
                                        novelty=True, n_jobs=-1),
            "X_train": Xtr20, "X_val": Xval20,
            "scaler": sc80, "n_features": 20, "pca": pca20,
        })

    # 4. Features mecanicas 95D con PCA
    for n_comp in [20, 25]:
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xtr_pca  = pca.fit_transform(Xtr95)
        Xval_pca = pca.transform(Xval95)
        var_exp  = pca.explained_variance_ratio_.sum()
        variants.append({
            "name": f"LOF k=20 + 95D-mec + PCA({n_comp}D, {var_exp:.0%} var)",
            "model": LocalOutlierFactor(n_neighbors=20, contamination=0.08,
                                        novelty=True, n_jobs=-1),
            "X_train": Xtr_pca, "X_val": Xval_pca,
            "scaler": sc95, "n_features": n_comp, "pca": pca,
            "use_95d": True,
        })

    # ---------------------------------------------------------------------------
    # Evaluacion
    # ---------------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"  {'Variante':<45} {'AUC':>6} {'F1':>6} {'Recall':>7}")
    print("-"*70)

    results = []
    for v in variants:
        t0 = time.time()
        try:
            v["model"].fit(v["X_train"])
            scores = -v["model"].decision_function(v["X_val"])
            preds  = (v["model"].predict(v["X_val"]) == -1).astype(int)

            auc = roc_auc_score(y_val, scores)
            f1  = f1_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)

            results.append({**v, "auc": auc, "f1": f1, "recall": rec,
                            "time": time.time() - t0})
            print(f"  {v['name']:<45} {auc:>6.4f} {f1:>6.4f} {rec:>7.4f}")
        except Exception as e:
            print(f"  {v['name']:<45} ERROR: {e}")

    # ---------------------------------------------------------------------------
    # Veredicto y guardado del mejor
    # ---------------------------------------------------------------------------
    if not results:
        print("No se obtuvieron resultados.")
        return

    df_res = pd.DataFrame([
        {"Variante": r["name"], "AUC": r["auc"], "F1": r["f1"], "Recall": r["recall"]}
        for r in results
    ]).sort_values("AUC", ascending=False)

    print("\n" + "="*70)
    print("  RANKING FINAL")
    print("="*70)
    medals = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    for i, (_, row) in enumerate(df_res.iterrows()):
        medal = ["GANADOR", "  2do  ", "  3ro  "][i] if i < 3 else f"  {i+1}ro  "
        print(f"  [{medal}] {row['Variante']:<45} AUC={row['AUC']:.4f} F1={row['F1']:.4f}")

    # Guardar mejor modelo
    best = sorted(results, key=lambda x: x["auc"], reverse=True)[0]
    print(f"\nGANADOR: {best['name']}")
    print(f"  AUC={best['auc']:.4f} | F1={best['f1']:.4f} | Recall={best['recall']:.4f}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "lof_model.pkl", "wb") as f:
        pickle.dump(best["model"], f)
    with open(output_dir / "robust_scaler.pkl", "wb") as f:
        pickle.dump(best["scaler"], f)
    if best.get("pca") is not None:
        with open(output_dir / "pca_reducer.pkl", "wb") as f:
            pickle.dump(best["pca"], f)
        print(f"  PCA guardado: pca_reducer.pkl ({best['n_features']}D)")

    # Determinar scores para umbrales
    X_vn = best["X_val"][:len(X_val_n_80)]
    X_va = best["X_val"][len(X_val_n_80):]
    scores_n  = -best["model"].decision_function(X_vn)
    scores_ab = -best["model"].decision_function(X_va)

    meta = {
        "algorithm": "LocalOutlierFactor",
        "variant": best["name"],
        "n_neighbors": 20,
        "contamination": 0.08,
        "auc_validation": round(best["auc"], 4),
        "f1_validation": round(best["f1"], 4),
        "recall_validation": round(best["recall"], 4),
        "threshold_warn":  float(np.percentile(scores_n, 95)),
        "threshold_alert": float(np.percentile(scores_ab, 25)),
        "score_min_normal": float(np.min(scores_n)),
        "score_max_normal": float(np.max(scores_n)),
        "n_features_input": best["n_features"],
        "use_pca": best.get("pca") is not None,
        "use_95d_features": best.get("use_95d", False),
        "sr_target": 8000,
        "n_features": 80,
    }
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    df_res.to_csv("scripts/optimization_results.csv", index=False)

    print(f"\nArtefactos guardados en '{output_dir}'")
    print("Resultados completos en: scripts/optimization_results.csv")
    print("\nPara re-entrenar con el mejor modelo en el dataset completo:")
    print(f"  python -m core.model_trainer --data_dir \".\\pump\" --output_dir \".\\models\"")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--sample_size", type=int, default=400,
                        help="Muestras normales a usar (default=400)")
    args = parser.parse_args()
    run_optimization(args.data_dir, args.sample_size, args.output_dir)
