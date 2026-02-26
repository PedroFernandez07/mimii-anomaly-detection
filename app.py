"""
app.py
======
Tablero de Control Industrial — Detección de Anomalías Acústicas
Sistema basado en MIMII Dataset (Pump, +6 dB SNR) | UNI Mecatrónica

Interfaz Streamlit que simula un SCADA de planta con 4 estaciones de bomba.

Ejecutar:
  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Importaciones del proyecto ────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core.inference_engine import InferenceEngine, InferenceResult, STATUS_NORMAL, STATUS_ALERT, STATUS_WARN

# ---------------------------------------------------------------------------
# Configuración de página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MIMII — Monitor Industrial de Bombas",
    page_icon="MMM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS personalizado — aspecto de panel de control industrial
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Fondo oscuro industrial */
.stApp { background-color: #0d1117; color: #c9d1d9; }
.main .block-container { padding-top: 1rem; max-width: 1400px; }

/* Cards de estación */
.station-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.4rem 0;
    transition: border-color 0.3s;
}
.station-card:hover { border-color: #58a6ff; }
.station-normal { border-left: 4px solid #3fb950; }
.station-warn   { border-left: 4px solid #d29922; }
.station-alert  { border-left: 4px solid #f85149; }
.station-idle   { border-left: 4px solid #8b949e; }

/* KPI badges */
.kpi-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}
.badge-normal { background:#1a3a1e; color:#3fb950; }
.badge-warn   { background:#3b2a0e; color:#d29922; }
.badge-alert  { background:#3b1214; color:#f85149; }
.badge-idle   { background:#1e2530; color:#8b949e; }

/* Métricas */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
}
[data-testid="metric-container"] label { color: #8b949e !important; font-size:0.75rem; }
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #58a6ff !important; font-size: 1.6rem !important; font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* Header */
h1 { color: #58a6ff !important; }
h2, h3 { color: #79c0ff !important; }

/* Gauge container */
.gauge-wrapper { display: flex; justify-content: center; }

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]
STATUS_CSS = {
    STATUS_NORMAL: ("normal", "badge-normal"),
    STATUS_WARN:   ("warn",   "badge-warn"),
    STATUS_ALERT:  ("alert",  "badge-alert"),
}
STATUS_LABEL_ES = {
    STATUS_NORMAL: "NORMAL",
    STATUS_WARN:   "ADVERTENCIA",
    STATUS_ALERT:  "ALERTA CRÍTICA",
}
MODEL_DIR = Path(__file__).parent / "models"


# ---------------------------------------------------------------------------
# Estado de sesión
# ---------------------------------------------------------------------------
def _init_session():
    if "station_results" not in st.session_state:
        st.session_state.station_results: dict[str, InferenceResult | None] = {
            mid: None for mid in MACHINE_IDS
        }
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "engine_error" not in st.session_state:
        st.session_state.engine_error = None


_init_session()


# ---------------------------------------------------------------------------
# Carga del modelo (cacheada)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de IA...")
def get_engine(model_dir: str) -> InferenceEngine | None:
    try:
        engine = InferenceEngine(model_dir=model_dir)
        engine._load()
        return engine
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Componentes de visualización
# ---------------------------------------------------------------------------

def render_health_gauge(health: float, size: int = 200) -> plt.Figure:
    """Gauge circular de Índice de Salud estilo industrial."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100),
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    # Color según salud
    if health >= 70:
        color = "#3fb950"
    elif health >= 40:
        color = "#d29922"
    else:
        color = "#f85149"

    # Fondo del arco
    theta_bg = np.linspace(0, np.pi, 100)
    ax.fill_between(theta_bg, 0.7, 1.0, color="#30363d", alpha=0.8)

    # Arco de salud (0° = mínimo, 180° = máximo)
    angle = np.pi * (health / 100.0)
    theta_fg = np.linspace(0, angle, 100)
    ax.fill_between(theta_fg, 0.7, 1.0, color=color, alpha=0.95)

    # Texto central
    ax.text(np.pi / 2, 0.35, f"{health:.0f}%",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=color, transform=ax.transData)
    ax.text(np.pi / 2, 0.1, "SALUD",
            ha="center", va="center", fontsize=8,
            color="#8b949e", transform=ax.transData)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, np.pi)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(1)
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    return fig


def render_feature_sparkline(features: np.ndarray) -> plt.Figure:
    """Mini gráfica de barras de los 80 features normalizados."""
    fig, ax = plt.subplots(figsize=(5, 1.2))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    norm = (features - features.min()) / (np.ptp(features) + 1e-9)
    colors = ["#3fb950" if v < 0.6 else "#d29922" if v < 0.85 else "#f85149"
              for v in norm]
    ax.bar(range(len(norm)), norm, color=colors, width=1.0, linewidth=0)
    ax.set_xlim(0, 80)
    ax.set_ylim(0, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def render_station_card(machine_id: str, result: InferenceResult | None) -> None:
    """Renderiza la card de una estación de bomba."""
    if result is None:
        css_class = "idle"
        badge_class = "badge-idle"
        status_text = "SIN DATOS"
        health_val = "—"
        score_val = "—"
    elif result.error:
        css_class = "alert"
        badge_class = "badge-alert"
        status_text = "ERROR"
        health_val = "—"
        score_val = "—"
    else:
        css_class = STATUS_CSS.get(result.status, ("idle", "badge-idle"))[0]
        badge_class = STATUS_CSS.get(result.status, ("idle", "badge-idle"))[1]
        status_text = STATUS_LABEL_ES.get(result.status, result.status)
        health_val = f"{result.health_index:.1f}%"
        score_val = f"{result.anomaly_score:.4f}"

    num = machine_id.split("_")[-1]
    st.markdown(f"""
    <div class="station-card station-{css_class}">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:1.1rem; font-weight:700; color:#c9d1d9;">
                    ⚙️ BOMBA {num}
                </span>
                <span style="font-size:0.75rem; color:#8b949e; margin-left:0.5rem;">
                    {machine_id}
                </span>
            </div>
            <span class="kpi-badge {badge_class}">{status_text}</span>
        </div>
        <div style="margin-top:0.8rem; display:flex; gap:1.5rem;">
            <div>
                <div style="font-size:0.7rem; color:#8b949e;">ÍNDICE SALUD</div>
                <div style="font-size:1.4rem; font-weight:700; color:#58a6ff;">{health_val}</div>
            </div>
            <div>
                <div style="font-size:0.7rem; color:#8b949e;">ANOMALY SCORE</div>
                <div style="font-size:1.4rem; font-weight:700; color:#79c0ff;">{score_val}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Configuración y carga
# ---------------------------------------------------------------------------

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## ⚙️ PANEL DE CONTROL")
        st.markdown("---")

        # Estado del modelo
        engine = get_engine(str(MODEL_DIR))
        if engine is not None and engine._loaded:
            st.success("✅ Modelo cargado")
            meta = engine._meta
            if meta:
                st.markdown(f"""
                <div style="font-size:0.78rem; color:#8b949e; line-height:1.8;">
                AUC Validación: <b style="color:#3fb950">{meta.get('auc_validation','—')}</b><br>
                Features: <b>{meta.get('n_features','80')}D</b> |
                Estimadores: <b>{meta.get('n_estimators','—')}</b><br>
                SR: <b>{meta.get('sr_target','8000')} Hz</b> |
                Nyquist: <b>4000 Hz</b>
                </div>""", unsafe_allow_html=True)
        else:
            st.error("⚠️ Modelo no encontrado")
            st.markdown("""
            **Para activar el sistema:**
            ```bash
            python core/model_trainer.py \\
              --data_dir /ruta/mimii_data \\
              --output_dir ./models
            ```
            """)

        st.markdown("---")
        st.markdown("### 📤 Cargar Audio de Prueba")

        selected_id = st.selectbox(
            "Estación objetivo:",
            MACHINE_IDS,
            format_func=lambda x: f"⚙️ Bomba {x.split('_')[-1]} ({x})",
        )

        uploaded = st.file_uploader(
            "Archivo WAV (.wav)",
            type=["wav"],
            help="Carga un archivo del MIMII Dataset para inferencia en tiempo real.",
        )

        ground_truth = st.selectbox(
            "Ground Truth (opcional):",
            ["—", "normal", "abnormal"],
            help="Etiqueta real del dataset para auditoría técnica.",
        )

        run_btn = st.button(
            "🔍 EJECUTAR ANÁLISIS",
            use_container_width=True,
            type="primary",
            disabled=(uploaded is None),
        )

        st.markdown("---")
        st.markdown("### 🔄 Acciones")
        if st.button("Limpiar resultados", use_container_width=True):
            for mid in MACHINE_IDS:
                st.session_state.station_results[mid] = None
            st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.72rem; color:#8b949e; text-align:center; line-height:1.6;">
        <b>MIMII Anomaly Detection MVP</b><br>
        Ing. Mecatrónica · UNI Perú<br>
        Isolation Forest | RobustScaler<br>
        80-dim · 8 kHz · +6dB SNR
        </div>""", unsafe_allow_html=True)

        return selected_id, uploaded, ground_truth, run_btn


# ---------------------------------------------------------------------------
# Panel de resultados detallados
# ---------------------------------------------------------------------------

def render_detail_panel(result: InferenceResult, ground_truth: str) -> None:
    """Renderiza el panel de análisis detallado tras una inferencia."""
    if result.error:
        st.error(f"❌ Error en análisis: {result.error}")
        return

    st.markdown("---")
    st.markdown(f"### 📊 Análisis — {result.machine_id} | `{result.file_name}`")

    # KPIs principales
    c1, c2, c3, c4 = st.columns(4)
    status_emoji = result.status_emoji
    with c1:
        st.metric("Estado", f"{status_emoji} {STATUS_LABEL_ES.get(result.status, result.status)}")
    with c2:
        st.metric("Índice de Salud", f"{result.health_index:.1f} %")
    with c3:
        st.metric("Anomaly Score", f"{result.anomaly_score:.5f}")
    with c4:
        model_pred = "Anómalo" if result.is_anomaly else "Normal"
        st.metric("Predicción modelo", model_pred)

    col_gauge, col_spark = st.columns([1, 2])

    with col_gauge:
        st.markdown("**Gauge de Salud**")
        fig_gauge = render_health_gauge(result.health_index)
        st.pyplot(fig_gauge, use_container_width=False)
        plt.close(fig_gauge)

    with col_spark:
        st.markdown("**Vector de Características (80D)**")
        fig_spark = render_feature_sparkline(result.feature_vector)
        st.pyplot(fig_spark, use_container_width=True)
        plt.close(fig_spark)

        st.markdown("""
        <div style="font-size:0.72rem; color:#8b949e; margin-top:0.3rem;">
        🟢 Normal &nbsp;|&nbsp; 🟡 Atípico &nbsp;|&nbsp; 🔴 Crítico
        &nbsp;&nbsp;|&nbsp;&nbsp; Bloques: Temporal · PSD/Welch · Onsets · MFCCs
        </div>""", unsafe_allow_html=True)

    # ── Auditoría técnica (expandible) ────────────────────────────────────────
    with st.expander("🔬 AUDITORÍA TÉCNICA — Ground Truth & Validación"):
        if ground_truth != "—":
            model_pred_bool = result.is_anomaly
            gt_bool = ground_truth == "abnormal"
            match = model_pred_bool == gt_bool
            verdict = "✅ CORRECTO" if match else "❌ INCORRECTO"
            gt_color = "#f85149" if gt_bool else "#3fb950"

            st.markdown(f"""
            | Campo | Valor |
            |---|---|
            | **Ground Truth (Dataset MIMII)** | <span style="color:{gt_color}; font-weight:700;">{ground_truth.upper()}</span> |
            | **Predicción del Modelo** | {model_pred} |
            | **Concordancia** | {verdict} |
            | **Anomaly Score crudo** | `{result.anomaly_score:.6f}` |
            | **Umbral ADVERTENCIA** | `{result.metadata.get('threshold_warn', 'N/A')}` |
            | **Umbral ALERTA** | `{result.metadata.get('threshold_alert', 'N/A')}` |
            """, unsafe_allow_html=True)

            if not match:
                if not model_pred_bool and gt_bool:
                    st.warning("⚠️ **Falso Negativo**: La bomba presenta anomalía real "
                               "pero el modelo la clasificó como normal. "
                               "Considera reducir el umbral de alerta.")
                else:
                    st.info("ℹ️ **Falso Positivo**: El modelo emitió alerta sobre "
                            "una muestra normal. El nivel de ruido de +6dB puede "
                            "elevar scores en señales normales con interferencia.")
        else:
            st.info("Selecciona el Ground Truth en el sidebar para activar la auditoría.")

        # Features breakdown
        st.markdown("**Descomposición del vector de features por bloque:**")
        blocks = {
            "Temporales (0-14)": result.feature_vector[:15],
            "PSD/Welch (15-39)": result.feature_vector[15:40],
            "Onsets (40-49)": result.feature_vector[40:50],
            "MFCCs (50-79)": result.feature_vector[50:80],
        }
        cols = st.columns(4)
        for col, (name, block) in zip(cols, blocks.items()):
            with col:
                st.markdown(f"*{name}*")
                st.markdown(f"μ = `{np.mean(block):.3f}`")
                st.markdown(f"σ = `{np.std(block):.3f}`")
                st.markdown(f"max = `{np.max(block):.3f}`")


# ---------------------------------------------------------------------------
# Vista principal del tablero
# ---------------------------------------------------------------------------

def render_dashboard_overview() -> None:
    """Grid de 4 estaciones + estadísticas globales."""
    results = st.session_state.station_results
    analyzed = [r for r in results.values() if r is not None and r.error is None]

    # KPIs globales
    st.markdown("### 📡 Estado Global de la Planta")
    g1, g2, g3, g4 = st.columns(4)

    n_alert = sum(1 for r in analyzed if r.status == STATUS_ALERT)
    n_warn = sum(1 for r in analyzed if r.status == STATUS_WARN)
    n_normal = sum(1 for r in analyzed if r.status == STATUS_NORMAL)
    avg_health = np.mean([r.health_index for r in analyzed]) if analyzed else 0.0

    with g1:
        st.metric("🔴 Alertas", n_alert, delta=None)
    with g2:
        st.metric("🟡 Advertencias", n_warn)
    with g3:
        st.metric("🟢 Normales", n_normal)
    with g4:
        st.metric("❤️ Salud Media", f"{avg_health:.1f}%" if analyzed else "—")

    st.markdown("---")
    st.markdown("### 🏭 Estaciones de Bombas")

    # Grid 2x2
    row1 = st.columns(2)
    row2 = st.columns(2)
    grid = [row1[0], row1[1], row2[0], row2[1]]

    for col_widget, machine_id in zip(grid, MACHINE_IDS):
        with col_widget:
            render_station_card(machine_id, results[machine_id])


# ---------------------------------------------------------------------------
# App principal
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
        <div style="font-size:2.5rem;">⚙️</div>
        <div>
            <h1 style="margin:0; font-size:1.8rem;">Sistema de Monitoreo Predictivo</h1>
            <p style="margin:0; color:#8b949e; font-size:0.9rem;">
                MIMII Dataset · Bombas Industriales · +6 dB SNR ·
                Isolation Forest · 80-dim features · 8 kHz
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    selected_id, uploaded, ground_truth, run_btn = render_sidebar()

    # Inferencia
    if run_btn and uploaded is not None:
        engine = get_engine(str(MODEL_DIR))
        if engine is None:
            st.error("❌ El modelo no está disponible. Entrena primero con `model_trainer.py`.")
        else:
            with st.spinner(f"🔍 Analizando {uploaded.name} en {selected_id}..."):
                # Guardar temporalmente
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                try:
                    result = engine.predict(tmp_path, machine_id=selected_id)
                    result.file_name = uploaded.name
                    st.session_state.station_results[selected_id] = result
                finally:
                    os.unlink(tmp_path)

            # Toast de resultado
            if result.error:
                st.error(f"Error: {result.error}")
            elif result.status == STATUS_ALERT:
                st.error(f"🚨 ALERTA CRÍTICA en {selected_id} — Salud: {result.health_index:.1f}%")
            elif result.status == STATUS_WARN:
                st.warning(f"⚠️ ADVERTENCIA en {selected_id} — Salud: {result.health_index:.1f}%")
            else:
                st.success(f"✅ NORMAL — {selected_id} — Salud: {result.health_index:.1f}%")

    # Tablero principal
    render_dashboard_overview()

    # Panel detallado del último resultado del selected_id
    last_result = st.session_state.station_results.get(selected_id)
    if last_result is not None:
        render_detail_panel(last_result, ground_truth)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:0.72rem; color:#8b949e; padding:1rem 0;">
    Proyecto de Egreso · Ingeniería Mecatrónica · UNI Perú &nbsp;|&nbsp;
    MIMII Dataset (Tanaka et al., 2019) &nbsp;|&nbsp;
    Paradigma: One-Class Novelty Detection &nbsp;|&nbsp;
    Arquitectura: Isolation Forest + RobustScaler &nbsp;|&nbsp;
    Cloud-ready: Azure Functions / GCP Cloud Run
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
