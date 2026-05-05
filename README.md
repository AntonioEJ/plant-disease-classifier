
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

Cada miembro trabaja en su propia rama personal y carpeta de experimentos:

| Miembro   | Rama      | Carpeta de resultados         |
|-----------|-----------|------------------------------|
| Antonio   | `antonio` | `experiments/antonio/`       |
| Alondra   | `alondra` | `experiments/alondra/`       |
| Paulo     | `paulo`   | `experiments/paulo/`         |
| Arlette   | `arlette` | `experiments/arlette/`       |

Cada quien puede organizar sus experimentos y notebooks como prefiera dentro de su carpeta.
Para compartir resultados finales, haz un Pull Request de tu rama personal a `develop` o `main`.

---

## 🏗️ Estructura del Repositorio (carpetas principales)


```
configs/        # Configuraciones YAML de experimentos
data/           # Datos (no se suben al repo)
experiments/    # Resultados de cada miembro y métricas
│
├── antonio/
│   ├── resnet50_warmup/
│   │   ├── metrics.json
│   │   ├── classification_report.txt
│   │   ├── model_summary.txt
│   │   └── figures/
│   ├── densenet121_warmup/
│   │   └── ...
│
├── alondra/
│   ├── resnet50_straightforward/
│   ├── vgg16_straightforward/
│   ├── vgg16_finetune/
│   └── ...
│
├── paulo/
│   ├── resnet50_finetune/
│   ├── densenet121_straightforward/
│   └── ...
│
├── arlette/
│   ├── densenet121_finetune/
│   ├── vgg16_warmup/
│   └── ...
│
├── all_results.csv           # Tabla resumen de todos los experimentos

notebooks/      # Análisis y comparación de resultados
reports/        # Figuras y reportes
```

---

## ⚙️ Criterios Base (Comparabilidad)

> ⚠️ **Estos parámetros son INVARIABLES.** Cambiarlos invalida la comparación entre experimentos.

| Criterio | Valor |
|---|---|
| Split | 70% train / 15% val / 15% test |
| Split strategy | Estratificado por clase |
| Tamaño de imagen | 224×224×3 |
uv run python train.py --model resnet50 --strategy finetune --member paulo
# 🌿 Plant Disease Classifier — Deep Learning

Clasificación binaria de enfermedades en hojas de plantas mediante transfer learning.
Dataset: [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

---

## 📋 Descripción breve

Proyecto colaborativo para detectar enfermedades en plantas usando deep learning. El flujo es 100% con notebooks de Python, sin scripts ni módulos extra.

---

## 📁 Estructura del repositorio

```
configs/        # Configuraciones YAML de experimentos
data/           # Datos locales (no se suben al repo)
experiments/    # Resultados y análisis de cada miembro
notebooks/      # Notebooks principales (EDA, split, entrenamiento, análisis)
reports/        # Figuras y reportes
```

---



## 🚦 Descarga de datos desde Kaggle

1. Instala [uv](https://astral.sh/uv/) si no lo tienes:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2. Instala la Kaggle CLI:
    ```bash
    uv pip install kaggle
    ```

3. Descarga tu archivo `kaggle.json` desde https://www.kaggle.com/settings
    y colócalo en `~/.kaggle/kaggle.json`:
    ```bash
    mkdir -p ~/.kaggle
    cp /ruta/al/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    ```

4. Descarga el dataset:
    ```bash
    uv run kaggle datasets download -d arjuntejaswi/plant-village -p data/raw/ --unzip
    # El directorio resultante debe ser: data/raw/PlantVillage/
    ```

---

## 🚦 ¿Cómo usar los notebooks?

1. Abre y ejecuta los notebooks en la carpeta `notebooks/`:
    - `split_dataset.ipynb`: genera los splits de train/val/test y los guarda en JSON.
    - `dataset_pipeline.ipynb`: funciones para cargar y preparar datos con tf.data.
    - Otros notebooks: experimentos, análisis, visualización.
2. Guarda tus resultados y análisis en tu carpeta dentro de `experiments/`.

---

## ⚙️ Criterios base para todos los experimentos

| Criterio         | Valor                                  |
|------------------|----------------------------------------|
| Split            | 70% train / 15% val / 15% test         |
| Estrategia split | Estratificado por clase                |
| Tamaño imagen    | 224×224×3                              |
| Preprocesamiento | `preprocess_input` del modelo          |
| Augmentation     | Solo en train                          |
| Cabeza           | GAP → Dense(128, relu) → Dropout(0.4) → Dense(1, sigmoid) |
| Loss             | binary_crossentropy                    |
| Class weights    | Sí (balanced)                          |
| Seed             | 42                                     |

---

## 🧑‍💻 Buenas prácticas

- Usa solo notebooks para todo el flujo (EDA, split, entrenamiento, análisis).
- Documenta tus experimentos y resultados en tu carpeta de `experiments/`.
- Si necesitas compartir splits, súbelos a `data/splits/` o usa Drive.
- No subas datos originales ni archivos pesados al repositorio.

---

## 📚 Notebooks principales

- `notebooks/split_dataset.ipynb`: Split reproducible y guardado en JSON.
- `notebooks/dataset_pipeline.ipynb`: Funciones para cargar/preparar datos.
- `notebooks/`: Agrega aquí tus experimentos y análisis.

---

## 📝 ¿Dudas?

Pregunta en el grupo o revisa los notebooks de ejemplo. El flujo es simple y todo se hace desde notebooks.
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

git clone https://github.com/antonioesparza/plant-disease-classifier.git

## 🔄 Reproducibilidad

Para reproducir exactamente un experimento (flujo 100% notebooks):

```bash
# 1. Clona el repositorio
git clone https://github.com/antonioesparza/plant-disease-classifier.git
cd plant-disease-classifier

# 2. Instala dependencias (usa uv o conda)
uv pip install -r requirements.txt
# o bien
conda env create -f environment.yml
conda activate plant-disease-dl

# 3. Descarga el dataset de Kaggle
uv run kaggle datasets download -d arjuntejaswi/plant-village -p data/raw/ --unzip

# 4. Copia el splits.json compartido (repo o Drive)
# cp /ruta/compartida/splits.json data/splits/

# 5. Abre y ejecuta los notebooks en Jupyter/VS Code
#   - notebooks/split_dataset.ipynb (opcional, si quieres regenerar splits)
#   - notebooks/dataset_pipeline.ipynb (carga/preprocesamiento)
#   - Tu notebook de experimento
```

El seed 42 está fijo en `configs/base_config.yaml` para asegurar reproducibilidad.

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
