"""
benchmark.py
============
Evaluación comparativa de algoritmos de Novelty Detection
para el MIMII Dataset (Pump, +6 dB SNR).

Algoritmos evaluados:
  1. Isolation Forest
  2. Local Outlier Factor (LOF)
  3. One-Class SVM
  4. Elliptic Envelope (Robust Covariance)

Uso:
  python scripts/benchmark.py --data_dir ".\pump"
"""

import sys
import os
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.feature_extractor import extract_features

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Inventario
# ---------------------------------------------------------------------------

def build_inventory(data_dir: Path) -> pd.DataFrame:
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
    df = pd.DataFrame(records)
    return df[df["label"] != "unknown"].reset_index(drop=True)


def extract_batch(paths, desc=""):
    results, valid_labels = [], []
    for i, (path, label) in enumerate(tqdm(paths, desc=desc)):
        try:
            results.append(extract_features(path))
            valid_labels.append(label)
        except Exception:
            pass
    return np.vstack(results).astype(np.float32), valid_labels


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(data_dir: str, sample_size: int = 300, random_state: int = 42):
    print("\n" + "="*70)
    print("  BENCHMARK — Algoritmos de Novelty Detection")
    print("  MIMII Dataset · Pump · +6 dB SNR")
    print("="*70)

    df = build_inventory(Path(data_dir))
    normal_df   = df[df["label"] == "normal"].sample(frac=1, random_state=random_state)
    abnormal_df = df[df["label"] == "abnormal"].sample(frac=1, random_state=random_state)

    # Limitar tamaño para velocidad
    n_normal   = min(sample_size, len(normal_df))
    n_abnormal = min(150, len(abnormal_df))

    n_train = int(n_normal * 0.80)
    train_paths  = [(r, "normal") for r in normal_df["path"].iloc[:n_train].tolist()]
    val_n_paths  = [(r, "normal") for r in normal_df["path"].iloc[n_train:n_normal].tolist()]
    val_ab_paths = [(r, "abnormal") for r in abnormal_df["path"].iloc[:n_abnormal].tolist()]

    print(f"\n📦 Muestras → Train: {n_train} | Val Normal: {len(val_n_paths)} | Val Anómalo: {n_abnormal}")
    print("\n⏳ Extrayendo features (esto toma ~3-5 min)...\n")

    X_train, _   = extract_batch(train_paths,  desc="  Train (normal)")
    X_val_n, _   = extract_batch(val_n_paths,  desc="  Val   (normal)")
    X_val_ab, _  = extract_batch(val_ab_paths, desc="  Val   (abnormal)")

    # Normalización
    scaler = RobustScaler()
    X_train_sc  = scaler.fit_transform(X_train)
    X_val_n_sc  = scaler.transform(X_val_n)
    X_val_ab_sc = scaler.transform(X_val_ab)

    X_val   = np.vstack([X_val_n_sc, X_val_ab_sc])
    y_val   = np.array([0]*len(X_val_n_sc) + [1]*len(X_val_ab_sc))

    # PCA para One-Class SVM (reduce dimensión y acelera)
    pca = PCA(n_components=20, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val)

    # ---------------------------------------------------------------------------
    # Algoritmos
    # ---------------------------------------------------------------------------
    algorithms = {
        "Isolation Forest": {
            "model": IsolationForest(
                n_estimators=200, contamination=0.02,
                random_state=random_state, n_jobs=-1
            ),
            "X_train": X_train_sc,
            "X_val":   X_val,
        },
        "LOF (k=20)": {
            "model": LocalOutlierFactor(
                n_neighbors=20, contamination=0.08,
                novelty=True, n_jobs=-1
            ),
            "X_train": X_train_sc,
            "X_val":   X_val,
        },
        "LOF (k=5)": {
            "model": LocalOutlierFactor(
                n_neighbors=5, contamination=0.08,
                novelty=True, n_jobs=-1
            ),
            "X_train": X_train_sc,
            "X_val":   X_val,
        },
        "One-Class SVM": {
            "model": OneClassSVM(
                kernel="rbf", nu=0.05, gamma="scale"
            ),
            "X_train": X_train_pca,
            "X_val":   X_val_pca,
        },
        "Elliptic Envelope": {
            "model": EllipticEnvelope(
                contamination=0.05, random_state=random_state
            ),
            "X_train": X_train_pca,
            "X_val":   X_val_pca,
        },
    }

    results = []
    print("\n" + "="*70)
    print("  RESULTADOS")
    print("="*70)
    print(f"\n{'Algoritmo':<22} {'AUC':>6} {'F1':>6} {'Recall':>8} {'Precision':>10} {'Tiempo':>8}")
    print("-"*70)

    for name, cfg in algorithms.items():
        model     = cfg["model"]
        X_tr      = cfg["X_train"]
        X_v       = cfg["X_val"]

        t0 = time.time()
        try:
            model.fit(X_tr)
            scores = -model.decision_function(X_v)
            preds  = (model.predict(X_v) == -1).astype(int)
            elapsed = time.time() - t0

            auc  = roc_auc_score(y_val, scores)
            f1   = f1_score(y_val, preds, zero_division=0)
            rec  = recall_score(y_val, preds, zero_division=0)
            prec = precision_score(y_val, preds, zero_division=0)

            results.append({
                "Algoritmo": name, "AUC": auc, "F1": f1,
                "Recall": rec, "Precision": prec, "Tiempo_s": elapsed
            })
            print(f"{name:<22} {auc:>6.4f} {f1:>6.4f} {rec:>8.4f} {prec:>10.4f} {elapsed:>7.1f}s")

        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    # ---------------------------------------------------------------------------
    # Veredicto
    # ---------------------------------------------------------------------------
    if results:
        df_res = pd.DataFrame(results).sort_values("AUC", ascending=False)

        print("\n" + "="*70)
        print("  VEREDICTO — Ranking por AUC")
        print("="*70)
        for i, row in df_res.iterrows():
            medal = ["🥇", "🥈", "🥉", "4️⃣ ", "5️⃣ "][list(df_res.index).index(i)]
            print(f"  {medal} {row['Algoritmo']:<22} AUC={row['AUC']:.4f} | "
                  f"F1={row['F1']:.4f} | Recall={row['Recall']:.4f}")

        winner = df_res.iloc[0]["Algoritmo"]
        winner_auc = df_res.iloc[0]["AUC"]
        print(f"\n✅ GANADOR: {winner} (AUC={winner_auc:.4f})")
        print("\n💡 Recomendación:")

        if "LOF" in winner:
            print("   Considera reemplazar Isolation Forest por LOF en el pipeline.")
            print("   Actualiza MODEL_CLASS en inference_engine.py")
        elif "SVM" in winner:
            print("   One-Class SVM con PCA(20D) da mejor separación en este dataset.")
        elif "Elliptic" in winner:
            print("   El dataset tiene distribución gaussiana aprovechable.")
        else:
            print("   Isolation Forest sigue siendo la mejor opción. Ajusta umbrales.")

        print("\n" + "="*70)
        df_res.to_csv("scripts/benchmark_results.csv", index=False)
        print("📄 Resultados guardados en: scripts/benchmark_results.csv")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_size", type=int, default=300,
                        help="Muestras normales a usar (default 300, más rápido que el total)")
    args = parser.parse_args()
    run_benchmark(args.data_dir, args.sample_size)