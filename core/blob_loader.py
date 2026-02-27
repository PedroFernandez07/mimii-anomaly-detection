"""
blob_loader.py
==============
Descarga los artefactos del modelo desde Azure Blob Storage
si no estan disponibles localmente.

Comportamiento:
  - En local: usa los .pkl del directorio models/ directamente
  - En Azure Container Apps: descarga desde Blob Storage al arrancar

Variables de entorno requeridas en produccion:
  AZURE_STORAGE_CONNECTION_STRING
  AZURE_STORAGE_CONTAINER (default: models)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CONTAINER_NAME    = os.getenv("AZURE_STORAGE_CONTAINER", "models")
CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

ARTIFACTS = [
    "lof_model.pkl",
    "robust_scaler.pkl",
    "training_meta.json",
]


def download_models_if_needed(model_dir: str | Path = "./models") -> bool:
    """
    Descarga los artefactos desde Azure Blob Storage si no existen localmente.
    Retorna True si los modelos estan disponibles, False si falla.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    missing = [a for a in ARTIFACTS if not (model_dir / a).exists()]
    if not missing:
        logger.info("Modelos disponibles localmente en '%s'", model_dir)
        return True

    if not CONNECTION_STRING:
        logger.warning(
            "Modelos no encontrados y AZURE_STORAGE_CONNECTION_STRING no configurado."
        )
        return False

    try:
        from azure.storage.blob import BlobServiceClient

        logger.info("Descargando modelos desde Azure Blob Storage...")
        client    = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container = client.get_container_client(CONTAINER_NAME)

        for artifact in missing:
            dest = model_dir / artifact
            with open(dest, "wb") as f:
                container.get_blob_client(artifact).download_blob().readinto(f)
            logger.info("  OK %s descargado", artifact)

        logger.info("Todos los modelos descargados correctamente.")
        return True

    except ImportError:
        logger.error("Instala: pip install azure-storage-blob")
        return False
    except Exception as exc:
        logger.error("Error descargando modelos: %s", exc)
        return False
