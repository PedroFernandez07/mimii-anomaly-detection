"""
recalibrate.py
==============
Recalibra los umbrales del tablero sin reentrenar el modelo.
Lee los scores reales del modelo sobre el dataset y calcula
umbrales correctos para NORMAL / ADVERTENCIA / ALERTA.

Uso:
  python recalibrate.py --data_dir ".\pump"
"""

import sys, os, json, pickle, warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from core.feature_extractor import extract_features

warnings.filterwarnings("ignore")

def recalibrate(data_dir: str, model_dir: str = "./models",
                n_samples: int = 200):

    model_dir = Path(model_dir)

    # Cargar artefactos
    with open(model_dir / "lof_model.pkl", "rb") as f:
        lof = pickle.load(f)
    with open(model_dir / "robust_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(model_dir / "training_meta.json", "r") as f:
        meta = json.load(f)

    # Recolectar archivos
    normal_paths, abnormal_paths = [], []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith(".wav"):
                continue
            parts = Path(root).parts
            if "normal" in parts:
                normal_paths.append(str(Path(root) / f))
            elif "abnormal" in parts:
                abnormal_paths.append(str(Path(root) / f))

    # Muestrear
    rng = np.random.default_rng(42)
    normal_sample   = rng.choice(normal_paths,
                                  min(n_samples, len(normal_paths)),
                                  replace=False).tolist()
    abnormal_sample = rng.choice(abnormal_paths,
                                  min(n_samples, len(abnormal_paths)),
                                  replace=False).tolist()

    def get_scores(paths, label):
        scores = []
        for p in tqdm(paths, desc=f"  scores {label}"):
            try:
                feat = extract_features(p)
                X = scaler.transform(feat.reshape(1, -1))
                scores.append(float(-lof.decision_function(X)[0]))
            except Exception:
                pass
        return np.array(scores)

    print("\nCalculando scores reales del modelo...")
    scores_normal   = get_scores(normal_sample,   "normal")
    scores_abnormal = get_scores(abnormal_sample, "abnormal")

    print(f"\nDistribucion de scores NORMALES:")
    print(f"  min={scores_normal.min():.4f} | p25={np.percentile(scores_normal,25):.4f} "
          f"| p50={np.percentile(scores_normal,50):.4f} "
          f"| p75={np.percentile(scores_normal,75):.4f} "
          f"| p90={np.percentile(scores_normal,90):.4f} "
          f"| max={scores_normal.max():.4f}")

    print(f"\nDistribucion de scores ANOMALOS:")
    print(f"  min={scores_abnormal.min():.4f} | p25={np.percentile(scores_abnormal,25):.4f} "
          f"| p50={np.percentile(scores_abnormal,50):.4f} "
          f"| p75={np.percentile(scores_abnormal,75):.4f} "
          f"| max={scores_abnormal.max():.4f}")

    # Calibracion correcta
    # threshold_warn  = donde el 90% de los normales estan por debajo
    # threshold_alert = donde el 75% de los anomalos estan por encima
    thresh_warn  = float(np.percentile(scores_normal,   90))
    thresh_alert = float(np.percentile(scores_abnormal, 25))

    print(f"\nUmbrales calibrados:")
    print(f"  threshold_warn  = {thresh_warn:.6f}  (p90 de normales)")
    print(f"  threshold_alert = {thresh_alert:.6f}  (p25 de anomalos)")

    # Verificar solapamiento
    overlap = np.sum(scores_normal > thresh_warn) / len(scores_normal)
    detection = np.sum(scores_abnormal >= thresh_alert) / len(scores_abnormal)
    print(f"\n  Falsos positivos esperados: {overlap:.1%}")
    print(f"  Deteccion de anomalos:      {detection:.1%}")

    # Actualizar meta
    meta["score_min_normal"] = float(np.min(scores_normal))
    meta["score_max_normal"] = float(np.max(scores_normal))
    meta["threshold_warn"]   = thresh_warn
    meta["threshold_alert"]  = thresh_alert

    with open(model_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\ntraining_meta.json actualizado correctamente.")
    print("Reinicia Streamlit para aplicar los cambios: streamlit run app.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--model_dir",  default="./models")
    parser.add_argument("--n_samples",  type=int, default=200)
    args = parser.parse_args()
    recalibrate(args.data_dir, args.model_dir, args.n_samples)
