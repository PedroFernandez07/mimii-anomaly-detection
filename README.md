<div align="center">

# MIMII Anomaly Detection

### Deteccion de Anomalias Acusticas en Bombas Industriales

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Azure](https://img.shields.io/badge/Azure-Container_Apps-0078D4?style=flat-square&logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com)

**Curso de Inteligencia Artificial · Prof. Ing. Ivan Calle**
**Ingenieria Mecatronica (9no ciclo) · Universidad Nacional de Ingenieria (UNI) · Lima, Peru**

[Demo](#demo) · [Instalacion](#instalacion) · [Arquitectura](#arquitectura) · [Resultados](#resultados)

</div>

---

## Descripcion

Sistema de mantenimiento predictivo basado en deteccion de anomalias acusticas para bombas centrifugas industriales. Implementado sobre el benchmark **MIMII Dataset** (Malfunctioning Industrial Machine Investigation and Inspection) bajo condiciones reales de ruido industrial (+6 dB SNR).

El pipeline abarca desde la ingesta de audio crudo hasta el despliegue en la nube, incluyendo extraccion manual de features DSP, seleccion empirica de algoritmos mediante benchmarking, optimizacion de umbral para contexto industrial y dashboard de monitoreo en tiempo real.

---

## Arquitectura End-to-End

![Arquitectura del Sistema](docs/architecture.svg)

El pipeline esta desacoplado en tres capas:

- **Ingesta y Feature Engineering** — procesamiento de senales DSP en 80 dimensiones
- **Modelo ML** — Local Outlier Factor con umbral calibrado para industria
- **Interfaz y Cloud** — Streamlit Dashboard + microservicio REST

---

## Stack Tecnologico

| Capa | Tecnologia | Proposito |
|------|-----------|-----------|
| Lenguaje | Python 3.11 | Pipeline completo |
| Feature Engineering | librosa · scipy · numpy | Extraccion DSP |
| ML | scikit-learn | LOF · RobustScaler |
| Data | pandas | Inventario y manipulacion |
| UI | Streamlit | Dashboard de monitoreo |
| Cloud | Azure Container Apps · Blob Storage | Despliegue y almacenamiento |
| Versionamiento | GitHub | Control de versiones |

---

## Estructura del Proyecto

```
mimii-anomaly-detection/
├── app.py                          # Monitor acustico individual
├── central_monitoreo.py            # Central de monitoreo 2 plantas / 20 bombas
├── recalibrate.py                  # Recalibracion de umbrales
├── Dockerfile                      # Contenedor app.py (Azure)
├── Dockerfile.central              # Contenedor central_monitoreo.py (Azure)
├── requirements.txt
│
├── core/
│   ├── feature_extractor.py        # Pipeline 80D (DSP)
│   ├── model_trainer.py            # Entrenamiento LOF
│   ├── inference_engine.py         # Motor de inferencia cloud-ready
│   └── blob_loader.py              # Descarga de modelos desde Azure Blob Storage
│
├── scripts/
│   ├── benchmark.py                # Comparacion 5 algoritmos
│   └── cloud_handler.py            # Handler Azure Functions / GCP
│
├── models/
│   └── training_meta.json          # Metadatos del modelo entrenado
│
└── docs/
    └── architecture.svg            # Diagrama de arquitectura
```

> Los artefactos `lof_model.pkl` y `robust_scaler.pkl` no estan en el repositorio. En produccion se descargan automaticamente desde Azure Blob Storage. Para desarrollo local, generarlos con los pasos 4 y 5 de la instalacion.

---

## Pipeline de Features — Vector 80D

Descriptores acusticos extraidos manualmente, especializados para maquinaria rotativa, procesados a 8,000 Hz (Nyquist = 4,000 Hz):

| Bloque | Dims | Descriptores |
|--------|------|-------------|
| Temporales | 15D | RMS, Crest Factor, Kurtosis, Skewness, ZCR, Shape/Impulse/Margin Factor, IQR, Energy |
| PSD / Welch | 25D | 10 bandas 0-4kHz, centroide espectral, ancho de banda, flatness, entropia, frecuencia pico |
| Onsets | 10D | Tasa de impactos, IOI mean/std, entropia de envolvente, regularidad ritmica |
| MFCCs + Delta | 30D | 13 coeficientes MFCC + 9 delta-1 + 8 delta-2 |

---

## Benchmark Comparativo

Se evaluaron 5 algoritmos sobre datos reales antes de seleccionar el modelo. Las metricas de esta tabla corresponden al umbral por defecto de scikit-learn sobre un subconjunto de 300 muestras — su proposito es comparar la capacidad discriminativa de cada algoritmo, no el rendimiento final del sistema.

| Ranking | Algoritmo | AUC | F1 | Recall |
|---------|-----------|-----|----|--------|
| 1 | **LOF k=20** | **0.906** | **0.756** | 0.620 |
| 2 | LOF k=5 | 0.887 | 0.778 | 0.653 |
| 3 | One-Class SVM | 0.748 | 0.573 | 0.407 |
| 4 | Isolation Forest | 0.763 | 0.159 | 0.087 |
| 5 | Elliptic Envelope | 0.653 | 0.000 | 0.000 |

LOF k=20 fue seleccionado por su mayor AUC y su capacidad de modelar densidades locales, caracteristica ventajosa para señales acusticas de maquinaria donde las anomalias son cambios sutiles en el patron de vibracion.

---

## Resultados Finales

Tras entrenar con el dataset completo y optimizar el umbral de decision para maximizar Recall con la restriccion Precision >= 0.55:

| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| AUC-ROC | 0.891 | Alta capacidad discriminativa |
| Recall | 0.901 | 9 de cada 10 fallas detectadas |
| Precision | 0.558 | Trade-off aceptado para contexto industrial |
| F1-Score | 0.690 | Balance global |
| Escenario | +6 dB SNR | Condicion de mayor ruido del dataset |

**Nota sobre la diferencia entre benchmark y resultado final:** El Recall del benchmark (0.620 para LOF k=20) y el Recall del modelo final (0.901) corresponden a etapas distintas. El benchmark usa el umbral por defecto de scikit-learn para comparar algoritmos. El modelo final usa un umbral optimizado mediante busqueda sobre el rango de percentiles del conjunto de validacion, priorizando la deteccion de fallas sobre la precision. El algoritmo es el mismo; lo que cambia es la posicion del corte de decision.

El umbral fue calibrado para Recall >= 0.90 porque en mantenimiento industrial el costo de una falla no detectada supera al de una inspeccion innecesaria. Por cada falla real detectada se generan aproximadamente 0.8 falsas alarmas — ratio operacionalmente aceptable.

---

## Instalacion

```bash
# 1. Clonar repositorio
git clone https://github.com/PedroFernandez07/mimii-anomaly-detection.git
cd mimii-anomaly-detection

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar dataset
# MIMII Dataset (6_dB_pump.zip) desde https://zenodo.org/record/3384388
# Extraer en ./pump/

# 4. Entrenar modelo
python -m core.model_trainer --data_dir "./pump" --output_dir "./models"

# 5. Recalibrar umbrales
python recalibrate.py --data_dir "./pump"

# 6. Lanzar monitor individual
streamlit run app.py

# 7. Lanzar central de monitoreo (2 plantas / 20 bombas)
streamlit run central_monitoreo.py
```

---

## Demo

**Live demo (Azure Container Apps):**

| Interfaz | URL |
|----------|-----|
| Monitor individual | https://mimii-app.wonderfulfield-74501e25.eastus.azurecontainerapps.io/ |
| Central de monitoreo | https://mimii-central.wonderfulfield-74501e25.eastus.azurecontainerapps.io/ |

El sistema incluye dos interfaces:

**Monitor individual (`app.py`)** — carga un archivo WAV, selecciona el machine ID y ejecuta el analisis. Muestra estado, indice de salud, score, vector de features 80D y auditoria contra Ground Truth. Mantiene historial de hasta 20 analisis en la sesion.

**Central de monitoreo (`central_monitoreo.py`)** — simula dos plantas industriales con 20 bombas cada una. Selecciona archivos al azar del conjunto de validacion (datos no vistos durante el entrenamiento), ejecuta el analisis en lote y muestra el estado de cada bomba con accuracy en tiempo real contra Ground Truth.

---

## Despliegue en Azure

El motor de inferencia esta desacoplado de la UI y opera como microservicio:

```python
from core.inference_engine import predict

result = predict("pump_sound.wav", machine_id="id_00")
# {'status': 'NORMAL', 'health_index': 87.3, 'anomaly_score': -0.2341}
```

Infraestructura desplegada:

| Servicio | Recurso | Proposito |
|----------|---------|-----------|
| Azure Container Registry | mimiiregistry | Almacena imagenes Docker |
| Azure Container Apps | mimii-app | Monitor individual (app.py) |
| Azure Container Apps | mimii-central | Central de monitoreo (central_monitoreo.py) |
| Azure Blob Storage | mimiimodels / models | Artefactos del modelo (LOF + Scaler) |
| Azure Blob Storage | mimiimodels / audio-val | Audios de validacion del dataset |
| Log Analytics | workspace-mimiirg | Monitoreo y logs |

---

## Dataset

**MIMII Dataset** — Malfunctioning Industrial Machine Investigation and Inspection

- Fuente: Zenodo · [DOI: 10.5281/zenodo.3384388](https://zenodo.org/record/3384388)
- Subset: Pump · +6 dB SNR · 7.7 GB
- Distribucion: 3,749 normales · 456 anomalas · ratio 9:1
- Formato: WAV · 16 kHz · 16-bit PCM · ~10 segundos por muestra

> Purohit, H. et al. (2019). *MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection*. DCASE Workshop 2019.

---

## Autor

**Pedro Fernandez**
Estudiante de Ingenieria Mecatronica (9no ciclo) · Universidad Nacional de Ingenieria (UNI) · Lima, Peru
Curso: Inteligencia Artificial · Prof. Ing. Ivan Calle · Especializacion en curso: Data Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-pedro--fernandez--avila-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/pedro-fernandez-avila)
[![GitHub](https://img.shields.io/badge/GitHub-PedroFernandez07-181717?style=flat-square&logo=github)](https://github.com/PedroFernandez07)

---

<div align="center">
<sub>
Paradigma: One-Class Novelty Detection · Algoritmo: Local Outlier Factor · Stack: Python · scikit-learn · Streamlit · Azure
</sub>
</div>