# 🌿 Plant Disease Classifier — Deep Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://python.org)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Clasificación binaria de enfermedades en hojas de plantas mediante transfer learning.  
Dataset: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

---

## 📌 Descripción del Problema

| Aspecto | Detalle |
|---|---|
| Tipo | Clasificación binaria supervisada |
| Input | Imágenes RGB 224×224×3 |
| Clases | `0` = healthy, `1` = diseased |
| Métrica prioritaria | **Recall** (minimizar falsos negativos) |
| Objetivo de negocio | Detectar **todas** las plantas enfermas — un FN es más costoso que un FP |

---

## 👥 Equipo y Distribución de Experimentos

| Modelo | Estrategia | Responsable | Branch |
|---|---|---|---|
| ResNet50 | Straightforward | Antonio | `feature/antonio/resnet-straightforward` |
| ResNet50 | Fine-tuning | Paulo | `feature/paulo/resnet-finetune` |
| ResNet50 | Warm-up | Antonio | `feature/antonio/resnet-warmup` |
| DenseNet121 | Straightforward | Paulo | `feature/paulo/densenet-straightforward` |
| DenseNet121 | Fine-tuning | Arlette | `feature/arlette/densenet-finetune` |
| DenseNet121 | Warm-up | Antonio | `feature/antonio/densenet-warmup` |
| VGG16 | Straightforward | Antonio | `feature/antonio/vgg-straightforward` |
| VGG16 | Fine-tuning | Antonio | `feature/antonio/vgg-finetune` |
| VGG16 | Warm-up | Arlette | `feature/arlette/vgg-warmup` |

> **Alondra** apoya en integración, análisis de resultados y notebook de comparación.

---

## 🏗️ Estructura del Repositorio

```
plant-disease-classifier/
│
├── configs/                    # Configuraciones YAML de experimentos
│   ├── base_config.yaml        # ⚠️ Config base compartida — no modificar sin consenso
│   ├── models/
│   │   ├── resnet50.yaml
│   │   ├── densenet121.yaml
│   │   └── vgg16.yaml
│   └── strategies/
│       ├── straightforward.yaml
│       ├── finetune.yaml
│       └── warmup.yaml
│
├── data/                       # Datos (gitignored)
│   ├── raw/                    # Dataset original descargado de Kaggle
│   ├── processed/              # Datos procesados (si aplica)
│   └── splits/
│       └── splits.json         # ✅ COMPARTIR con el equipo — split único
│
├── src/                        # Código fuente
│   ├── data/
│   │   ├── dataset.py          # Pipeline tf.data, scan_plant_village
│   │   └── splits.py           # Splits estratificados reproducibles
│   ├── models/
│   │   ├── base_model.py       # Cabeza, freeze/unfreeze, compile
│   │   ├── resnet50.py
│   │   ├── densenet121.py
│   │   └── vgg16.py
│   ├── training/
│   │   ├── trainer.py          # Trainer multi-fase unificado
│   │   ├── callbacks.py        # Stack de callbacks estándar
│   │   └── class_weights.py    # Manejo de desbalance
│   ├── evaluation/
│   │   ├── metrics.py          # Evaluación completa + CSV maestro
│   │   └── visualize.py        # Plots de history y confusion matrix
│   └── utils/
│       ├── config.py           # Loader de configs YAML
│       ├── logger.py           # Logger estructurado
│       └── seed.py             # Reproducibilidad global
│
├── experiments/                # Outputs de experimentos (gitignored excepto CSV)
│   ├── antonio/
│   ├── alondra/
│   ├── paulo/
│   ├── arlette/
│   └── all_results.csv         # ✅ CSV maestro — agregar al repo tras cada experimento
│
├── notebooks/
│   └── results_comparison.ipynb  # Análisis final del equipo
│
├── reports/
│   ├── figures/                # Gráficas generadas
│   └── results/                # CSVs de resultados consolidados
│
├── tests/
│   ├── test_data.py
│   ├── test_metrics.py
│   └── test_models.py
│
├── .github/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/ci.yml
│
├── train.py                    # Punto de entrada principal
├── compare_results.py          # Agregación y comparación de resultados
├── requirements.txt
├── environment.yml
├── CONTRIBUTING.md
└── README.md
```

---

## ⚙️ Criterios Base (Comparabilidad)

> ⚠️ **Estos parámetros son INVARIABLES.** Cambiarlos invalida la comparación entre experimentos.

| Criterio | Valor |
|---|---|
| Split | 70% train / 15% val / 15% test |
| Split strategy | Estratificado por clase |
| Tamaño de imagen | 224×224×3 |
| Preprocesamiento | `preprocess_input` del modelo (NO `rescale=1./255`) |
| Augmentation | Solo en train |
| Cabeza | GAP → Dense(128, relu) → Dropout(0.4) → Dense(1, sigmoid) |
| Loss | binary_crossentropy |
| Class weights | Sí (balanced) |
| Seed | 42 |

---

## 🚀 Instalación

Este proyecto usa **[uv](https://docs.astral.sh/uv/)** como gestor de entorno y dependencias.

### Instalar uv (si no lo tienes)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clonar e instalar
```bash
git clone <repo_url>
cd plant-disease-classifier

# Crear virtualenv e instalar dependencias de producción + desarrollo
uv sync --extra dev

# (Opcional) Incluir también las herramientas de tracking MLflow/TensorBoard
uv sync --extra dev --extra tracking
```

El virtualenv queda en `.venv/` dentro del repositorio. No necesitas activarlo manualmente.

### Configurar Kaggle API
```bash
# 1. Descargar kaggle.json desde https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 📥 Descarga del Dataset

```bash
uv run kaggle datasets download -d arjuntejaswi/plant-village \
    -p data/raw/ --unzip
# El directorio resultante debe ser: data/raw/PlantVillage/
```

---

## 🔀 Estrategia de Branching

```
main          ← Estable, solo acepta merge desde develop con resultados finales
  └── develop ← Integración continua, todos los PRs apuntan aquí
        ├── feature/antonio/resnet-straightforward
        ├── feature/antonio/resnet-warmup
        ├── feature/antonio/densenet-warmup
        ├── feature/antonio/vgg-straightforward
        ├── feature/antonio/vgg-finetune
        ├── feature/paulo/resnet-finetune
        ├── feature/paulo/densenet-straightforward
        ├── feature/arlette/densenet-finetune
        └── feature/arlette/vgg-warmup
```

### Crear tu branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/<tu-nombre>/<modelo>-<estrategia>
# Ejemplo:
git checkout -b feature/antonio/resnet-warmup
```

---

## 🏋️ Correr un Experimento

### Paso 1 — Generar splits (solo la primera vez, UN solo miembro)
```bash
uv run python train.py --model resnet50 --strategy straightforward \
    --member antonio --data_dir data/raw/PlantVillage
# Esto genera data/splits/splits.json — COMPARTIR con el equipo via Git LFS o Drive
```

### Paso 2 — Entrenar (todos los demás usan el splits.json existente)
```bash
# Ejemplos de cada miembro:

# Antonio — ResNet50 Straightforward
uv run python train.py --model resnet50 --strategy straightforward --member antonio

# Paulo — ResNet50 Fine-tuning
uv run python train.py --model resnet50 --strategy finetune --member paulo

# Arlette — DenseNet121 Fine-tuning
uv run python train.py --model densenet121 --strategy finetune --member arlette

# Antonio — VGG16 Warm-up
uv run python train.py --model vgg16 --strategy warmup --member antonio
```

### Paso 3 — Ver TensorBoard
```bash
uv run tensorboard --logdir experiments/
```

### Paso 4 — Comparar resultados del equipo
```bash
uv run python compare_results.py
```

---

## 🔬 Estrategias de Transfer Learning

### Straightforward
El backbone queda completamente congelado. Solo se entrena la cabeza.
```
[ImageNet weights] → FROZEN backbone → HEAD (trainable)
```
**Cuándo funciona bien:** cuando las imágenes son similares a ImageNet.

### Fine-tuning
Fase 1: entrenar la cabeza. Fase 2: descongelar las últimas N capas.
```
Fase 1: FROZEN backbone → HEAD (trainable) → 10 épocas, lr=1e-3
Fase 2: PARTIAL backbone → HEAD (trainable) → 40 épocas, lr=1e-5
```
**Cuándo funciona bien:** con suficientes datos y dominio similar.

### Warm-up (descongelamiento progresivo)
```
Fase 1: FROZEN backbone → HEAD (10 épocas, lr=1e-3)
Fase 2: Últimas 50 capas descongeladas (15 épocas, lr=1e-4)
Fase 3: Backbone completo (25 épocas, lr=1e-5)
```
**Cuándo funciona bien:** dominio distante a ImageNet o datasets pequeños.

---

## 📊 Métricas

| Métrica | Prioridad | Por qué |
|---|---|---|
| **Recall** | 🔴 ALTA | Detectar todas las plantas enfermas. Un FN = planta enferma que se propaga. |
| F1-Score | 🟡 MEDIA | Balance entre Recall y Precision |
| Precision | 🟡 MEDIA | Evitar sobrealarmar con FP |
| Accuracy | 🟢 BAJA | Puede ser engañosa con desbalance de clases |

---

## ✅ Flujo de Pull Request

1. Terminar el experimento y verificar que genera todos los outputs
2. Correr los tests: `pytest tests/ -v`
3. Copiar `metrics.json` al directorio correspondiente
4. Hacer commit incluyendo: código + metrics.json + figures/
5. Abrir PR hacia `develop` usando el template de PR
6. Solicitar review de otro miembro del equipo

---

## 🧪 Tests

```bash
# Todos los tests
uv run pytest tests/ -v

# Solo tests de datos
uv run pytest tests/test_data.py -v

# Solo tests de modelos (requiere TensorFlow)
uv run pytest tests/test_models.py -v

# Con cobertura
uv run pytest tests/ --cov=src --cov-report=term-missing
```

---

## 📂 Outputs por Experimento

Cada experimento genera en `experiments/<member>/<model>_<strategy>/`:

```
resnet50_warmup/
├── history.json                    # History completo de todas las fases
├── metrics.json                    # Métricas finales (recall, precision, f1, accuracy)
├── confusion_matrix.npy            # Array NumPy de la confusion matrix
├── classification_report.txt       # Reporte completo por clase
├── model_summary.txt               # Arquitectura del modelo
├── resnet50_warmup_p1_best.keras   # Mejor modelo fase 1 (por val_recall)
├── resnet50_warmup_p2_best.keras   # Mejor modelo fase 2
├── resnet50_warmup_p3_best.keras   # Mejor modelo fase 3
├── resnet50_warmup_p*_epoch_log.csv
├── tensorboard/
└── figures/
    ├── resnet50_warmup_history.png
    └── resnet50_warmup_confusion_matrix.png
```

---

## 🔄 Reproducibilidad

Para reproducir exactamente un experimento:

```bash
# 1. Clonar el repositorio
git clone <repo_url>
cd plant-disease-classifier

# 2. Instalar dependencias
conda env create -f environment.yml
conda activate plant-disease-dl

# 3. Descargar el dataset
kaggle datasets download -d arjuntejaswi/plant-village -p data/raw/ --unzip

# 4. Obtener el splits.json del equipo (via repo o Drive)
# cp /ruta/compartida/splits.json data/splits/

# 5. Correr el experimento
python train.py --model resnet50 --strategy warmup --member antonio
```

El seed 42 está fijo en `configs/base_config.yaml` y en `src/utils/seed.py`.

---

## 🛠️ Herramientas de MLOps

| Herramienta | Propósito | Comando |
|---|---|---|
| TensorBoard | Visualizar entrenamiento en tiempo real | `tensorboard --logdir experiments/` |
| MLflow (opcional) | Tracking de experimentos centralizado | `mlflow ui` |
| pytest | Tests de pipeline | `pytest tests/ -v` |
| Black | Formateo de código | `black src/ tests/` |

---

## ⚠️ Errores Comunes y Cómo Evitarlos

| ❌ Error | ✅ Solución |
|---|---|
| Usar `rescale=1./255` con `preprocess_input` | Usar SOLO `preprocess_input` del modelo |
| Generar splits distintos por miembro | Compartir `data/splits/splits.json` y usar `load_splits()` |
| Aplicar augmentation a val/test | `augment=False` en `build_dataset()` para val y test |
| Olvidar compilar después de unfreeze | `compile_model()` se llama automáticamente en cada fase del Trainer |
| Monitorear `val_accuracy` en EarlyStopping | El monitor es `val_recall` (ver `callbacks.py`) |
| Ignorar class_weights | `use_class_weights: true` en `base_config.yaml` |
