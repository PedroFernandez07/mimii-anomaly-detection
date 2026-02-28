"""
app.py
======
Monitor de Anomalias Acusticas — MIMII Pump Dataset
Curso de Inteligencia Artificial · Prof. Ing. Ivan Calle
Ingenieria Mecatronica · UNI
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from collections import deque

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.inference_engine import InferenceEngine, InferenceResult, STATUS_NORMAL, STATUS_ALERT, STATUS_WARN
from core.blob_loader import download_models_if_needed

MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
MODEL_DIR   = Path(__file__).parent / "models"
MAX_HISTORY = 20

st.set_page_config(
    page_title="MIMII Acoustic Monitor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — mismo sistema de diseño que central_monitoreo.py
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
  background: #1a1a1a;
  color: #c8c8c8;
  font-family: 'IBM Plex Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── TOPBAR ── */
.topbar {
  background: #111;
  border: 1px solid #2e2e2e;
  border-radius: 4px;
  padding: 0 24px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}
.topbar-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  color: #e0e0e0;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.topbar-sep { width: 1px; height: 20px; background: #2e2e2e; margin: 0 20px; display:inline-block; }
.topbar-sub { font-size: 11px; color: #555; }
.topbar-right {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #444;
  text-align: right;
  line-height: 1.8;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: #111 !important;
  border-right: 1px solid #2a2a2a !important;
}
section[data-testid="stSidebar"] * { color: #c8c8c8 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #1a1a1a !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 2px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 12px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] {
  background: #161616 !important;
  border: 1px dashed #2e2e2e !important;
  border-radius: 2px !important;
}

/* ── STATUS INDICATOR ── */
.status-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 18px;
  border-radius: 2px;
  border: 1px solid #2a2a2a;
  margin-bottom: 16px;
}
.status-row.s-normal { border-left: 3px solid #27ae60; background: #0d1a10; }
.status-row.s-alert  { border-left: 3px solid #c0392b; background: #1a0d0d; }
.status-row.s-warn   { border-left: 3px solid #b7770d; background: #1a150d; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.dot-normal { background: #27ae60; }
.dot-alert  { background: #c0392b; }
.dot-warn   { background: #b7770d; }
.status-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.c-normal { color: #27ae60; }
.c-alert  { color: #c0392b; }
.c-warn   { color: #b7770d; }
.status-machine {
  margin-left: auto;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #555;
  letter-spacing: 1px;
}

/* ── METRICS STRIP ── */
.metrics-strip {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1px;
  background: #2a2a2a;
  border: 1px solid #2a2a2a;
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 20px;
}
.metric-block {
  background: #1a1a1a;
  padding: 12px 16px;
}
.metric-lbl {
  font-size: 9px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  margin-bottom: 4px;
}
.metric-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 20px;
  font-weight: 500;
  color: #c8c8c8;
}
.metric-val.v-ok   { color: #27ae60; }
.metric-val.v-warn { color: #b7770d; }
.metric-val.v-bad  { color: #c0392b; }
.metric-val.v-blue { color: #5b8dd9; }

/* ── HEALTH BAR ── */
.health-bar-container {
  margin-bottom: 20px;
}
.health-bar-label {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 6px;
}
.health-bar-bg {
  height: 6px;
  background: #2a2a2a;
  border-radius: 1px;
  overflow: hidden;
}
.health-bar-fill {
  height: 6px;
  border-radius: 1px;
  transition: width 0.5s ease;
}

/* ── FEATURE CHART CONTAINER ── */
.chart-section {
  background: #161616;
  border: 1px solid #2a2a2a;
  border-radius: 2px;
  padding: 16px;
  margin-bottom: 20px;
}
.chart-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 10px;
}

/* ── GT AUDIT PANEL ── */
.audit-panel {
  background: #161616;
  border: 1px solid #2a2a2a;
  border-radius: 2px;
  padding: 16px;
  margin-bottom: 20px;
}
.audit-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid #2a2a2a;
}
.audit-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid #212121;
  font-size: 12px;
}
.audit-key { color: #555; font-size: 11px; }
.audit-val {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  color: #c8c8c8;
}
.audit-ok  { color: #27ae60; }
.audit-err { color: #c0392b; }

/* ── HISTORY TABLE ── */
.history-section {
  margin-top: 24px;
}
.history-title {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #555;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 1px;
}
.history-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
.history-table th {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: #444;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  padding: 8px 12px;
  text-align: left;
  border-bottom: 1px solid #2a2a2a;
  background: #111;
}
.history-table td {
  padding: 8px 12px;
  border-bottom: 1px solid #1e1e1e;
  font-family: 'IBM Plex Mono', monospace;
  color: #888;
  vertical-align: middle;
}
.history-table tr:hover td { background: #1e1e1e; }

.h-dot { width: 6px; height: 6px; border-radius: 50%; display:inline-block; margin-right: 6px; vertical-align: middle; }
.h-normal { background: #27ae60; }
.h-alert  { background: #c0392b; }
.h-warn   { background: #b7770d; }

.verdict-ok  { color: #27ae60; }
.verdict-err { color: #c0392b; }

/* ── SIDEBAR MODEL STATUS ── */
.model-status {
  background: #0d1a10;
  border: 1px solid #1a3020;
  border-radius: 2px;
  padding: 10px 14px;
  margin-bottom: 16px;
}
.model-status.error {
  background: #1a0d0d;
  border-color: #3a1515;
}
.model-status-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 6px;
}
.model-status-label.ok  { color: #27ae60; }
.model-status-label.err { color: #c0392b; }
.model-meta {
  font-size: 11px;
  color: #555;
  line-height: 1.8;
}
.model-meta span { color: #888; }

/* Sidebar buttons */
section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  background: #1e1e1e !important;
  color: #c8c8c8 !important;
  border: 1px solid #2e2e2e !important;
  border-radius: 2px !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 11px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.5px !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  background: #252525 !important;
  border-color: #3a3a3a !important;
}

/* Sidebar divider */
.sidebar-section {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 9px;
  color: #333;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  padding: 12px 0 8px 0;
  border-top: 1px solid #1e1e1e;
  margin-top: 8px;
}

/* Footer */
.app-footer {
  border-top: 1px solid #2a2a2a;
  padding: 12px 0;
  display: flex;
  justify-content: space-between;
  margin-top: 32px;
}
.footer-text {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  color: #333;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: deque = deque(maxlen=MAX_HISTORY)
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_gt" not in st.session_state:
    st.session_state.last_gt = "—"


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine(model_dir: str) -> InferenceEngine | None:
    download_models_if_needed(model_dir)
    try:
        engine = InferenceEngine(model_dir=model_dir)
        engine._load()
        return engine
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers visualización
# ─────────────────────────────────────────────────────────────────────────────
def health_color(h: float) -> str:
    if h >= 60: return "#27ae60"
    if h >= 30: return "#b7770d"
    return "#c0392b"


def health_class(h: float) -> str:
    if h >= 60: return "v-ok"
    if h >= 30: return "v-warn"
    return "v-bad"


def status_dot_class(status: str) -> str:
    return {"NORMAL": "dot-normal", "ALERTA": "dot-alert", "ADVERTENCIA": "dot-warn"}.get(status, "dot-none")


def status_card_class(status: str) -> str:
    return {"NORMAL": "s-normal", "ALERTA": "s-alert", "ADVERTENCIA": "s-warn"}.get(status, "")


def status_label_class(status: str) -> str:
    return {"NORMAL": "c-normal", "ALERTA": "c-alert", "ADVERTENCIA": "c-warn"}.get(status, "")


def render_feature_chart(features: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 1.4))
    fig.patch.set_facecolor("#161616")
    ax.set_facecolor("#161616")
    norm = (features - features.min()) / (np.ptp(features) + 1e-9)
    colors = ["#27ae60" if v < 0.6 else "#b7770d" if v < 0.85 else "#c0392b" for v in norm]
    ax.bar(range(len(norm)), norm, color=colors, width=0.9, linewidth=0)
    # Block separators
    for sep in [15, 40, 50]:
        ax.axvline(sep, color="#2a2a2a", linewidth=1.5, zorder=5)
    ax.set_xlim(-0.5, 80.5)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


def render_gauge(health: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(2.4, 1.4), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#161616")
    ax.set_facecolor("#161616")
    color = health_color(health)
    theta_bg = np.linspace(0, np.pi, 200)
    ax.fill_between(theta_bg, 0.65, 1.0, color="#2a2a2a")
    angle = np.pi * (health / 100.0)
    theta_fg = np.linspace(0, angle, 200)
    ax.fill_between(theta_fg, 0.65, 1.0, color=color, alpha=0.9)
    ax.text(np.pi / 2, 0.30, f"{health:.0f}%",
            ha="center", va="center", fontsize=16, fontweight="500",
            color=color, transform=ax.transData,
            fontfamily="IBM Plex Mono")
    ax.set_ylim(0, 1); ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(1)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 16px 0 8px 0;">
          <div style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                      color:#888; letter-spacing:2px; text-transform:uppercase;">
            Acoustic Monitor
          </div>
          <div style="font-size:10px; color:#444; margin-top:4px;">
            MIMII Dataset · Pump · +6 dB SNR
          </div>
        </div>
        """, unsafe_allow_html=True)

        engine = get_engine(str(MODEL_DIR))

        if engine and engine._loaded:
            meta = engine._meta or {}
            st.markdown(f"""
            <div class="model-status">
              <div class="model-status-label ok">Model Loaded</div>
              <div class="model-meta">
                AUC <span>{meta.get('auc_validation','0.891')}</span> &nbsp;&middot;&nbsp;
                Recall <span>0.901</span><br>
                Features <span>{meta.get('n_features','80')}D</span> &nbsp;&middot;&nbsp;
                SR <span>{meta.get('sr_target','8000')} Hz</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-status error">
              <div class="model-status-label err">Model Not Found</div>
              <div class="model-meta">Ejecuta model_trainer.py</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Configuracion</div>', unsafe_allow_html=True)

        selected_id = st.selectbox(
            "Machine ID",
            MACHINE_IDS,
            format_func=lambda x: x,
            label_visibility="visible",
        )

        ground_truth = st.selectbox(
            "Ground Truth",
            ["—", "normal", "abnormal"],
            label_visibility="visible",
        )

        uploaded = st.file_uploader(
            "Audio WAV",
            type=["wav"],
            accept_multiple_files=False,
            label_visibility="visible",
        )

        run_btn = st.button(
            "Ejecutar analisis",
            use_container_width=True,
            disabled=(uploaded is None or engine is None),
        )

        st.markdown('<div class="sidebar-section">Sesion</div>', unsafe_allow_html=True)

        if st.button("Limpiar historial", use_container_width=True):
            st.session_state.history.clear()
            st.session_state.last_result = None
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:16px; font-family:'IBM Plex Mono',monospace;
                    font-size:10px; color:#333; line-height:2;">
          Local Outlier Factor<br>
          n_neighbors = 20<br>
          contamination = 0.08<br>
          RobustScaler · novelty=True
        </div>
        """, unsafe_allow_html=True)

        return selected_id, ground_truth, uploaded, run_btn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    selected_id, ground_truth, uploaded, run_btn = render_sidebar()
    engine = get_engine(str(MODEL_DIR))

    # Topbar
    st.markdown("""
    <div class="topbar">
      <div style="display:flex; align-items:center;">
        <span class="topbar-logo">MIMII Acoustic Monitor</span>
        <span class="topbar-sep"></span>
        <span class="topbar-sub">
          Curso de Inteligencia Artificial &nbsp;&middot;&nbsp;
          Prof. Ing. Ivan Calle &nbsp;&middot;&nbsp;
          Ingenieria Mecatronica &nbsp;&middot;&nbsp; UNI
        </span>
      </div>
      <div class="topbar-right">
        LOF &nbsp;&middot;&nbsp; AUC 0.891 &nbsp;&middot;&nbsp; Recall 0.901<br>
        One-Class Novelty Detection
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Inferencia ────────────────────────────────────────────────────────────
    if run_btn and uploaded and engine:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            with st.spinner(""):
                result = engine.predict(tmp_path, machine_id=selected_id)
                result.file_name = uploaded.name
        finally:
            os.unlink(tmp_path)

        st.session_state.last_result = result
        st.session_state.last_gt = ground_truth
        st.session_state.history.appendleft({
            "file": uploaded.name,
            "machine": selected_id,
            "status": result.status,
            "health": result.health_index,
            "score": result.anomaly_score,
            "gt": ground_truth,
            "correct": (
                (ground_truth == "abnormal") == result.is_anomaly
                if ground_truth != "—" else None
            ),
        })
        st.rerun()

    result: InferenceResult | None = st.session_state.last_result
    gt = st.session_state.last_gt

    # ── Panel resultado ───────────────────────────────────────────────────────
    if result and not result.error:
        status   = result.status
        health   = result.health_index
        score    = result.anomaly_score
        hc       = health_color(health)
        hcls     = health_class(health)
        slc      = status_label_class(status)
        scc      = status_card_class(status)
        dot_cls  = status_dot_class(status)

        # Status row
        st.markdown(f"""
        <div class="status-row {scc}">
          <span class="status-dot {dot_cls}"></span>
          <span class="status-label {slc}">{status}</span>
          <span class="status-machine">{result.machine_id} &nbsp;&middot;&nbsp; {result.file_name}</span>
        </div>
        """, unsafe_allow_html=True)

        # Metrics strip
        pred_str = "ANOMALO" if result.is_anomaly else "NORMAL"
        st.markdown(f"""
        <div class="metrics-strip">
          <div class="metric-block">
            <div class="metric-lbl">Health Index</div>
            <div class="metric-val {hcls}">{health:.1f}%</div>
          </div>
          <div class="metric-block">
            <div class="metric-lbl">Anomaly Score</div>
            <div class="metric-val {hcls}">{score:+.5f}</div>
          </div>
          <div class="metric-block">
            <div class="metric-lbl">Prediccion</div>
            <div class="metric-val" style="font-size:14px;">{pred_str}</div>
          </div>
          <div class="metric-block">
            <div class="metric-lbl">Machine ID</div>
            <div class="metric-val v-blue" style="font-size:14px;">{result.machine_id}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Health bar
        st.markdown(f"""
        <div class="health-bar-container">
          <div class="health-bar-label">
            <span>Health Index</span>
            <span>{health:.1f}%</span>
          </div>
          <div class="health-bar-bg">
            <div class="health-bar-fill" style="width:{health}%; background:{hc};"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Charts
        col_gauge, col_features = st.columns([1, 3])
        with col_gauge:
            st.markdown('<div class="chart-section"><div class="chart-title">Health Gauge</div>', unsafe_allow_html=True)
            fig_g = render_gauge(health)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_features:
            st.markdown('<div class="chart-section"><div class="chart-title">Feature Vector 80D — Temporal | PSD/Welch | Onsets | MFCCs</div>', unsafe_allow_html=True)
            fig_f = render_feature_chart(result.feature_vector)
            st.pyplot(fig_f, use_container_width=True)
            plt.close(fig_f)
            st.markdown("""
            <div style="font-size:10px; color:#333; margin-top:6px; font-family:'IBM Plex Mono',monospace;">
              0&#x2013;14: Temporal &nbsp;&nbsp; 15&#x2013;39: PSD/Welch &nbsp;&nbsp;
              40&#x2013;49: Onsets &nbsp;&nbsp; 50&#x2013;79: MFCCs+&Delta;
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Audit panel
        if gt != "—":
            gt_anom   = gt == "abnormal"
            correct   = gt_anom == result.is_anomaly
            gt_str    = "ANOMALO" if gt_anom else "NORMAL"
            verdict   = "CORRECTO" if correct else "INCORRECTO"
            v_cls     = "audit-ok" if correct else "audit-err"
            thresh_w  = result.metadata.get("threshold_warn", "—")
            thresh_a  = result.metadata.get("threshold_alert", "—")

            st.markdown(f"""
            <div class="audit-panel">
              <div class="audit-title">Auditoria — Ground Truth vs Prediccion</div>
              <div class="audit-row">
                <span class="audit-key">Ground Truth (MIMII Dataset)</span>
                <span class="audit-val">{gt_str}</span>
              </div>
              <div class="audit-row">
                <span class="audit-key">Prediccion del modelo</span>
                <span class="audit-val">{pred_str}</span>
              </div>
              <div class="audit-row">
                <span class="audit-key">Concordancia</span>
                <span class="audit-val {v_cls}">{verdict}</span>
              </div>
              <div class="audit-row">
                <span class="audit-key">Score crudo</span>
                <span class="audit-val">{score:.6f}</span>
              </div>
              <div class="audit-row">
                <span class="audit-key">Umbral advertencia</span>
                <span class="audit-val">{thresh_w}</span>
              </div>
              <div class="audit-row" style="border-bottom:none;">
                <span class="audit-key">Umbral alerta</span>
                <span class="audit-val">{thresh_a}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    elif result and result.error:
        st.markdown(f"""
        <div class="status-row s-alert">
          <span class="status-dot dot-alert"></span>
          <span class="status-label c-alert">Error</span>
          <span class="status-machine">{result.error}</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="padding: 48px 0; text-align:center; color:#333;
                    font-family:'IBM Plex Mono',monospace; font-size:12px;
                    letter-spacing:1px; text-transform:uppercase;">
          Cargue un archivo WAV y presione Ejecutar analisis
        </div>
        """, unsafe_allow_html=True)

    # ── Historial ─────────────────────────────────────────────────────────────
    if st.session_state.history:
        rows_html = ""
        for i, h in enumerate(st.session_state.history):
            dot_h = {"NORMAL": "h-normal", "ALERTA": "h-alert", "ADVERTENCIA": "h-warn"}.get(h["status"], "")
            if h["correct"] is None:
                verdict_html = '<span style="color:#333">—</span>'
            elif h["correct"]:
                verdict_html = '<span class="verdict-ok">&#10003;</span>'
            else:
                verdict_html = '<span class="verdict-err">&#10007;</span>'
            gt_disp = h["gt"] if h["gt"] != "—" else "—"
            rows_html += f"""
            <tr>
              <td style="color:#444">{i+1}</td>
              <td>{h['file']}</td>
              <td style="color:#666">{h['machine']}</td>
              <td><span class="h-dot {dot_h}"></span>{h['status']}</td>
              <td>{h['health']:.1f}%</td>
              <td>{h['score']:+.4f}</td>
              <td style="color:#555">{gt_disp}</td>
              <td>{verdict_html}</td>
            </tr>"""

        st.markdown(f"""
        <div class="history-section">
          <div class="history-title">Historial de sesion — {len(st.session_state.history)} registros</div>
          <table class="history-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Archivo</th>
                <th>Machine</th>
                <th>Estado</th>
                <th>Health</th>
                <th>Score</th>
                <th>Ground Truth</th>
                <th>Correcto</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="app-footer">
      <span class="footer-text">
        MIMII Dataset &nbsp;&middot;&nbsp; Pump &nbsp;&middot;&nbsp; +6 dB SNR &nbsp;&middot;&nbsp;
        Local Outlier Factor &nbsp;&middot;&nbsp; 80-dim &nbsp;&middot;&nbsp; SR 8 kHz
      </span>
      <span class="footer-text">
        Universidad Nacional de Ingenieria &nbsp;&middot;&nbsp; Ingenieria Mecatronica
      </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()