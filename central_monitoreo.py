"""
central_monitoreo.py
====================
Central de Monitoreo Industrial — MIMII Pump Dataset
Curso de Inteligencia Artificial · Prof. Ing. Ivan Calle
Ingenieria Mecatronica · UNI
"""

from __future__ import annotations
import os, random, tempfile
from pathlib import Path
import numpy as np
import streamlit as st
import sys
sys.path.insert(0, str(Path(__file__).parent))
from core.inference_engine import InferenceEngine
from core.blob_loader import download_models_if_needed

# ── Intentar importar azure-storage-blob ─────────────────────────────────────
try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

DATA_DIR         = Path("pump")           # fallback local
MODEL_DIR        = Path("models")
AUDIO_CONTAINER  = "audio-val"
MACHINE_IDS      = ["id_00", "id_02", "id_04", "id_06"]
BOMBAS_POR_PLANTA = 20
TRAIN_RATIO       = 0.80
RANDOM_STATE      = 42

st.set_page_config(
    page_title="MIMII Industrial Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background: #1a1a1a;
  color: #c8c8c8;
  font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }
div[data-testid="stToolbar"] { display: none; }

.app-wrapper { padding: 0; min-height: 100vh; background: #1a1a1a; }

.topbar {
  background: #111;
  border-bottom: 1px solid #2e2e2e;
  padding: 0 32px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.topbar-left { display: flex; align-items: center; gap: 24px; }
.topbar-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: #e0e0e0;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.topbar-sep { width: 1px; height: 20px; background: #2e2e2e; }
.topbar-sub { font-size: 11px; color: #666; font-weight: 300; }
.topbar-right {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #555;
  text-align: right;
  line-height: 1.6;
}

.toolbar {
  background: #161616;
  border-bottom: 1px solid #2a2a2a;
  padding: 10px 32px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.statsbar {
  background: #161616;
  border-bottom: 1px solid #2a2a2a;
  padding: 10px 32px;
  display: flex;
  align-items: center;
  gap: 32px;
}
.stat-block { display: flex; align-items: baseline; gap: 6px; }
.stat-num { font-family: 'IBM Plex Mono', monospace; font-size: 22px; font-weight: 500; }
.stat-lbl { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-divider { width: 1px; height: 28px; background: #2a2a2a; }
.c-alert  { color: #c0392b; }
.c-warn   { color: #b7770d; }
.c-ok     { color: #27ae60; }
.c-dim    { color: #555; }
.c-blue   { color: #5b8dd9; }
.c-white  { color: #e0e0e0; }

.pump-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: #2a2a2a;
  padding: 1px;
  margin: 0;
}

.pump-card {
  background: #1e1e1e;
  padding: 14px 16px;
  position: relative;
  transition: background 0.15s;
}
.pump-card:hover { background: #222; }
.pump-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 3px;
}
.pump-card.s-normal::before  { background: #27ae60; }
.pump-card.s-alert::before   { background: #c0392b; }
.pump-card.s-warn::before    { background: #b7770d; }
.pump-card.s-none::before    { background: #2e2e2e; }

.pump-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 8px;
}
.pump-id {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #555;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.pump-name {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #888;
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 160px;
}
.status-badge {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  padding: 2px 7px;
  border-radius: 2px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: inline-flex;
  align-items: center;
  gap: 5px;
}
.badge-normal  { background: #0d2918; color: #27ae60; }
.badge-alert   { background: #2a0f0d; color: #c0392b; }
.badge-warn    { background: #2a1f0d; color: #b7770d; }
.badge-none    { background: #1e1e1e; color: #444; border: 1px solid #2e2e2e; }

.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.dot-normal { background: #27ae60; }
.dot-alert  { background: #c0392b; }
.dot-warn   { background: #b7770d; }
.dot-none   { background: #333; }

.metrics-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; margin-top: 10px; }
.metric-cell {
  background: #181818;
  border: 1px solid #252525;
  padding: 6px 8px;
  border-radius: 2px;
}
.metric-lbl { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 3px; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 500; color: #c8c8c8; }
.metric-val.v-ok   { color: #27ae60; }
.metric-val.v-warn { color: #b7770d; }
.metric-val.v-bad  { color: #c0392b; }

.health-bar-bg { height: 2px; background: #2a2a2a; border-radius: 1px; margin-top: 4px; overflow: hidden; }
.health-bar-fill { height: 2px; border-radius: 1px; transition: width 0.4s ease; }

.gt-row {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid #252525;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.gt-lbl { font-size: 9px; color: #444; text-transform: uppercase; letter-spacing: 0.5px; }
.gt-val { font-family: 'IBM Plex Mono', monospace; font-size: 10px; display: flex; align-items: center; gap: 4px; }
.gt-ok  { color: #27ae60; }
.gt-err { color: #c0392b; }

.empty-state { grid-column: 1 / -1; padding: 80px 32px; text-align: center; color: #444; }
.empty-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.empty-sub { font-size: 12px; color: #333; }

.app-footer {
  border-top: 1px solid #2a2a2a;
  padding: 12px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 1px;
}
.footer-text { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #444; letter-spacing: 0.5px; }

div[data-testid="stButton"] > button {
  background: #252525 !important;
  color: #c8c8c8 !important;
  border: 1px solid #333 !important;
  border-radius: 2px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  font-weight: 400 !important;
  letter-spacing: 0.5px !important;
  padding: 6px 16px !important;
  text-transform: uppercase !important;
  transition: all 0.15s !important;
  height: auto !important;
}
div[data-testid="stButton"] > button:hover {
  background: #2e2e2e !important;
  border-color: #444 !important;
  color: #e0e0e0 !important;
}
div[data-testid="stButton"] > button:active { background: #333 !important; }

div[data-testid="stSelectbox"] > div > div {
  background: #1e1e1e !important;
  border: 1px solid #333 !important;
  border-radius: 2px !important;
  color: #c8c8c8 !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
}

div[data-testid="stProgressBar"] > div { background: #252525 !important; }
div[data-testid="stProgressBar"] > div > div { background: #5b8dd9 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Carga de catalogo de audios (Blob o local)
# ─────────────────────────────────────────────────────────────────────────────

def _get_connection_string() -> str | None:
    return os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


@st.cache_resource(show_spinner=False)
def get_blob_service() -> "BlobServiceClient | None":
    if not AZURE_AVAILABLE:
        return None
    conn = _get_connection_string()
    if not conn:
        return None
    try:
        return BlobServiceClient.from_connection_string(conn)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_blob_catalog() -> list[dict]:
    """Lista todos los blobs en audio-val y construye el catalogo."""
    service = get_blob_service()
    if service is None:
        return []
    try:
        container = service.get_container_client(AUDIO_CONTAINER)
        catalog = []
        for blob in container.list_blobs():
            # Estructura: {machine_id}/{label}/{filename}
            parts = blob.name.split("/")
            if len(parts) != 3:
                continue
            machine_id, label, filename = parts
            if machine_id not in MACHINE_IDS:
                continue
            if label not in ("normal", "abnormal"):
                continue
            catalog.append({
                "blob_name": blob.name,
                "machine_id": machine_id,
                "label": label,
                "filename": filename,
                "source": "blob",
            })
        return catalog
    except Exception:
        return []


def get_unseen_files_local(data_dir: Path) -> list[dict]:
    """Fallback: lee desde carpeta local pump/ con el mismo split."""
    unseen = []
    rng = np.random.default_rng(RANDOM_STATE)
    for machine_id in MACHINE_IDS:
        for label in ["normal", "abnormal"]:
            folder = data_dir / machine_id / label
            if not folder.exists():
                continue
            files = sorted(folder.glob("*.wav"))
            if not files:
                continue
            indices = rng.permutation(len(files))
            files_shuffled = [files[i] for i in indices]
            if label == "normal":
                n_train = int(len(files_shuffled) * TRAIN_RATIO)
                val_files = files_shuffled[n_train:]
            else:
                val_files = files_shuffled
            for f in val_files:
                unseen.append({
                    "path": str(f),
                    "machine_id": machine_id,
                    "label": label,
                    "filename": f.name,
                    "source": "local",
                })
    return unseen


@st.cache_data(show_spinner=False)
def get_all_unseen() -> list[dict]:
    """Devuelve catalogo desde Blob si esta disponible, si no desde local."""
    blob_catalog = build_blob_catalog()
    if blob_catalog:
        return blob_catalog
    # Fallback local
    return get_unseen_files_local(DATA_DIR)


def download_blob_to_tempfile(blob_name: str) -> str:
    """Descarga un blob a un archivo temporal y devuelve su ruta."""
    service = get_blob_service()
    if service is None:
        raise RuntimeError("Azure Blob no disponible")
    container = service.get_container_client(AUDIO_CONTAINER)
    blob_data = container.download_blob(blob_name).readall()
    suffix = Path(blob_name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(blob_data)
    tmp.flush()
    tmp.close()
    return tmp.name


def get_audio_path(bomba: dict) -> str:
    """Devuelve ruta local al audio (descargando desde Blob si necesario)."""
    if bomba.get("source") == "blob":
        return download_blob_to_tempfile(bomba["blob_name"])
    return bomba["path"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers UI
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_engine() -> InferenceEngine | None:
    download_models_if_needed(MODEL_DIR)
    try:
        engine = InferenceEngine(model_dir=MODEL_DIR)
        engine._load()
        return engine
    except Exception:
        return None


def status_badge(status: str) -> str:
    if status == "NORMAL":
        return '<span class="status-badge badge-normal"><span class="status-dot dot-normal"></span>NORMAL</span>'
    elif status == "ALERTA":
        return '<span class="status-badge badge-alert"><span class="status-dot dot-alert"></span>ALERTA</span>'
    elif status == "ADVERTENCIA":
        return '<span class="status-badge badge-warn"><span class="status-dot dot-warn"></span>ADVERTENCIA</span>'
    return '<span class="status-badge badge-none"><span class="status-dot dot-none"></span>SIN DATOS</span>'


def health_color(h: float) -> str:
    if h >= 60: return "#27ae60"
    if h >= 30: return "#b7770d"
    return "#c0392b"


def health_class(h: float) -> str:
    if h >= 60: return "v-ok"
    if h >= 30: return "v-warn"
    return "v-bad"


def card_class(status: str) -> str:
    return {"NORMAL": "s-normal", "ALERTA": "s-alert", "ADVERTENCIA": "s-warn"}.get(status, "s-none")


def render_card(idx: int, bomba: dict) -> str:
    result = bomba.get("result")
    mach   = bomba.get("machine_id", "—")
    fname  = bomba.get("filename", "—")

    if result is None:
        return f"""
        <div class="pump-card s-none">
          <div class="pump-header">
            <div>
              <div class="pump-id">PUMP {idx+1:02d} &middot; {mach}</div>
              <div class="pump-name">{fname}</div>
            </div>
            {status_badge("")}
          </div>
        </div>"""

    status    = result.get("status", "—")
    health    = result.get("health_index", 0.0)
    score     = result.get("anomaly_score", 0.0)
    gt        = bomba.get("label", "—")
    pred_anom = result.get("is_anomaly", False)
    gt_anom   = gt == "abnormal"
    correct   = gt_anom == pred_anom
    gt_str    = "ANOMALO" if gt_anom else "NORMAL"
    pred_str  = "ANOMALO" if pred_anom else "NORMAL"
    gt_cls    = "gt-ok" if correct else "gt-err"
    gt_icon   = "&#10003;" if correct else "&#10007;"
    hc        = health_color(health)
    hcls      = health_class(health)
    cc        = card_class(status)

    return f"""
    <div class="pump-card {cc}">
      <div class="pump-header">
        <div>
          <div class="pump-id">PUMP {idx+1:02d} &middot; {mach}</div>
          <div class="pump-name">{fname}</div>
        </div>
        {status_badge(status)}
      </div>
      <div class="metrics-row">
        <div class="metric-cell">
          <div class="metric-lbl">Health</div>
          <div class="metric-val {hcls}">{health:.0f}%</div>
          <div class="health-bar-bg">
            <div class="health-bar-fill" style="width:{health}%;background:{hc}"></div>
          </div>
        </div>
        <div class="metric-cell">
          <div class="metric-lbl">Score</div>
          <div class="metric-val {hcls}">{score:+.4f}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-lbl">Machine</div>
          <div class="metric-val" style="color:#888">{mach}</div>
        </div>
      </div>
      <div class="gt-row">
        <span class="gt-lbl">GT / Pred</span>
        <span class="gt-val {gt_cls}">{gt_icon} {gt_str} &rarr; {pred_str}</span>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for key, val in [
        ("bombas", {"Planta Norte": [], "Planta Sur": []}),
        ("planta", "Planta Norte"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    engine = get_engine()
    unseen = get_all_unseen()

    # ── TOPBAR ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="topbar">
      <div class="topbar-left">
        <span class="topbar-logo">MIMII Industrial Monitor</span>
        <span class="topbar-sep"></span>
        <span class="topbar-sub">
          Curso de Inteligencia Artificial &nbsp;&middot;&nbsp;
          Prof. Ing. Ivan Calle &nbsp;&middot;&nbsp;
          Ingenieria Mecatronica &nbsp;&middot;&nbsp; UNI
        </span>
      </div>
      <div class="topbar-right">
        LOF &nbsp;&middot;&nbsp; AUC 0.891 &nbsp;&middot;&nbsp; Recall 0.901<br>
        Validation set &mdash; datos no vistos en entrenamiento
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TOOLBAR ─────────────────────────────────────────────────────────────
    col_sel, col_load, col_run, col_clear = st.columns([3, 2, 2, 2])

    with col_sel:
        planta = st.selectbox(
            "Planta", ["Planta Norte", "Planta Sur"],
            key="planta", label_visibility="collapsed"
        )

    with col_load:
        cargar = st.button("Cargar seleccion aleatoria", use_container_width=True)

    with col_run:
        bombas_actuales = st.session_state.bombas.get(planta, [])
        analizar = st.button(
            "Ejecutar analisis",
            use_container_width=True,
            disabled=len(bombas_actuales) == 0
        )

    with col_clear:
        limpiar = st.button("Limpiar planta", use_container_width=True)

    # ── LOGICA BOTONES ───────────────────────────────────────────────────────
    if cargar:
        if not unseen:
            st.error("No se encontraron audios. Verifica que el container 'audio-val' en Azure Blob este poblado.")
        else:
            otra = "Planta Sur" if planta == "Planta Norte" else "Planta Norte"
            usados = {
                b.get("blob_name") or b.get("path")
                for b in st.session_state.bombas.get(otra, [])
            }
            disponibles = [
                f for f in unseen
                if (f.get("blob_name") or f.get("path")) not in usados
            ]
            if len(disponibles) < BOMBAS_POR_PLANTA:
                st.warning(f"Solo hay {len(disponibles)} archivos disponibles sin repetir con la otra planta.")
            seleccion = random.sample(disponibles, min(BOMBAS_POR_PLANTA, len(disponibles)))
            st.session_state.bombas[planta] = [{**b, "result": None} for b in seleccion]
            st.rerun()

    if limpiar:
        st.session_state.bombas[planta] = []
        st.rerun()

    bombas = st.session_state.bombas.get(planta, [])

    if analizar and bombas and engine:
        bar = st.progress(0)
        for i, bomba in enumerate(bombas):
            try:
                audio_path = get_audio_path(bomba)
                r = engine.predict(audio_path, machine_id=bomba["machine_id"])
                st.session_state.bombas[planta][i]["result"] = r.to_dict()
            except Exception as e:
                st.session_state.bombas[planta][i]["result"] = {
                    "status": "ALERTA",
                    "is_anomaly": True,
                    "health_index": 0.0,
                    "anomaly_score": 0.0,
                    "error": str(e),
                }
            finally:
                bar.progress((i + 1) / len(bombas))
        bar.empty()
        st.rerun()

    # ── STATSBAR ────────────────────────────────────────────────────────────
    bombas       = st.session_state.bombas.get(planta, [])
    analizadas   = [b for b in bombas if b.get("result")]
    alertas      = sum(1 for b in analizadas if b["result"]["status"] == "ALERTA")
    advertencias = sum(1 for b in analizadas if b["result"]["status"] == "ADVERTENCIA")
    normales     = sum(1 for b in analizadas if b["result"]["status"] == "NORMAL")
    sin_datos    = len(bombas) - len(analizadas)

    correctas = sum(
        1 for b in analizadas
        if (b["label"] == "abnormal") == b["result"]["is_anomaly"]
    )
    accuracy    = (correctas / len(analizadas) * 100) if analizadas else 0
    salud_media = np.mean([b["result"]["health_index"] for b in analizadas]) if analizadas else 0

    st.markdown(f"""
    <div class="statsbar">
      <div class="stat-block">
        <span class="stat-num c-alert">{alertas}</span>
        <span class="stat-lbl">Alert</span>
      </div>
      <div class="stat-block">
        <span class="stat-num c-warn">{advertencias}</span>
        <span class="stat-lbl">Warning</span>
      </div>
      <div class="stat-block">
        <span class="stat-num c-ok">{normales}</span>
        <span class="stat-lbl">Normal</span>
      </div>
      <div class="stat-block">
        <span class="stat-num c-dim">{sin_datos}</span>
        <span class="stat-lbl">Pending</span>
      </div>
      <div class="stat-divider"></div>
      <div class="stat-block">
        <span class="stat-num c-blue">{accuracy:.1f}%</span>
        <span class="stat-lbl">Accuracy</span>
      </div>
      <div class="stat-block">
        <span class="stat-num c-white">{salud_media:.1f}%</span>
        <span class="stat-lbl">Avg Health</span>
      </div>
      <div class="stat-divider"></div>
      <div style="margin-left:auto;font-family:'IBM Plex Mono',monospace;font-size:11px;color:#444">
        {planta.upper()} &nbsp;&middot;&nbsp; {len(analizadas)}/{len(bombas)} analizadas
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── GRID ────────────────────────────────────────────────────────────────
    if not bombas:
        st.markdown("""
        <div class="pump-grid">
          <div class="empty-state">
            <div class="empty-title">No hay datos cargados</div>
            <div class="empty-sub">Presione "Cargar seleccion aleatoria" para iniciar el monitoreo</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cards_html = "".join(render_card(i, b) for i, b in enumerate(bombas))
        st.markdown(f'<div class="pump-grid">{cards_html}</div>', unsafe_allow_html=True)

    # ── FOOTER ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
      <span class="footer-text">
        MIMII Dataset &nbsp;&middot;&nbsp; Pump &nbsp;&middot;&nbsp; +6 dB SNR
        &nbsp;&middot;&nbsp; Local Outlier Factor &nbsp;&middot;&nbsp; 80-dim features
        &nbsp;&middot;&nbsp; SR 8 kHz
      </span>
      <span class="footer-text">
        Universidad Nacional de Ingenieria &nbsp;&middot;&nbsp; Ingenieria Mecatronica
      </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__" or True:
    main()