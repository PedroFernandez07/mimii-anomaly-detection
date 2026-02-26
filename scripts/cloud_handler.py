"""
azure_function_handler.py
=========================
Handler de Azure Functions para el microservicio de inferencia.

Endpoint HTTP POST:
  Body JSON: { "audio_base64": "...", "machine_id": "id_00" }
  Response : { "status": "NORMAL", "health_index": 94.2, ... }

Deploy:
  func azure functionapp publish <app-name>
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile

# Azure Functions SDK (comentado si no está disponible)
# import azure.functions as func

from core.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# Instancia global — reutilizada entre invocaciones en el mismo contenedor
_engine: InferenceEngine | None = None


def _get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        _engine = InferenceEngine(model_dir=MODEL_DIR)
    return _engine


def predict_http(request_body: dict) -> dict:
    """
    Lógica de inferencia desacoplada del SDK de Azure/GCP.
    Puede ser invocada desde cualquier handler HTTP.

    Parameters
    ----------
    request_body : dict con claves:
        - audio_base64 (str) : Audio WAV codificado en base64
        - machine_id   (str) : ID de la bomba (id_00, id_02, id_04, id_06)

    Returns
    -------
    dict serializable a JSON con el resultado de inferencia
    """
    audio_b64 = request_body.get("audio_base64")
    machine_id = request_body.get("machine_id", "unknown")

    if not audio_b64:
        return {"error": "Campo 'audio_base64' requerido.", "status_code": 400}

    # Decodificar y guardar temporalmente
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as exc:
        return {"error": f"Error decodificando audio base64: {exc}", "status_code": 400}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        engine = _get_engine()
        result = engine.predict(tmp_path, machine_id=machine_id)
        response = result.to_dict()
        response["status_code"] = 200
    except Exception as exc:
        logger.exception("Error en inferencia")
        response = {"error": str(exc), "status_code": 500}
    finally:
        os.unlink(tmp_path)

    return response


# ── Azure Functions entry-point ───────────────────────────────────────────────
# Descomentar cuando se despliegue en Azure Functions:

# def main(req: func.HttpRequest) -> func.HttpResponse:
#     try:
#         body = req.get_json()
#     except Exception:
#         return func.HttpResponse(
#             json.dumps({"error": "JSON inválido"}),
#             mimetype="application/json", status_code=400
#         )
#
#     result = predict_http(body)
#     status_code = result.pop("status_code", 200)
#     return func.HttpResponse(
#         json.dumps(result),
#         mimetype="application/json",
#         status_code=status_code
#     )


# ── GCP Cloud Functions entry-point ──────────────────────────────────────────
# Descomentar cuando se despliegue en GCP:

# def gcp_handler(request):
#     import flask
#     body = request.get_json(silent=True) or {}
#     result = predict_http(body)
#     status_code = result.pop("status_code", 200)
#     return flask.jsonify(result), status_code
