# Guía de Contribución — Plant Disease Classifier

## Distribución de Experimentos

| Modelo | Estrategia | Responsable | Branch |
|---|---|---|---|
| ResNet50 | Straightforward | Alondra | `feature/alondra/resnet-straightforward` |
| ResNet50 | Fine-tuning | Paulo | `feature/paulo/resnet-finetune` |
| ResNet50 | Warm-up | Antonio | `feature/antonio/resnet-warmup` |
| DenseNet121 | Straightforward | Paulo | `feature/paulo/densenet-straightforward` |
| DenseNet121 | Fine-tuning | Arlette | `feature/arlette/densenet-finetune` |
| DenseNet121 | Warm-up | Antonio | `feature/antonio/densenet-warmup` |
| VGG16 | Straightforward | Alondra | `feature/alondra/vgg-straightforward` |
| VGG16 | Fine-tuning | Alondra | `feature/alondra/vgg-finetune` |
| VGG16 | Warm-up | Arlette | `feature/arlette/vgg-warmup` |

---

## Flujo de Trabajo Colaborativo

### 1. Setup inicial (una sola vez)

```bash
# Clonar el repositorio
git clone <repo_url>
cd plant-disease-classifier

# Instalar dependencias con uv (crea .venv automáticamente)
uv sync --extra dev

# Verificar instalación
uv run pytest tests/test_data.py tests/test_metrics.py -v
```

### 2. Iniciar un nuevo experimento

```bash
# Siempre partir desde develop actualizado
git checkout develop
git pull origin develop

# Crear tu branch (convención estricta)
# Formato: feature/<tu-nombre>/<modelo>-<estrategia>
# Ejemplos:
git checkout -b feature/alondra/resnet-straightforward  # Alondra
git checkout -b feature/antonio/resnet-warmup           # Antonio
git checkout -b feature/paulo/resnet-finetune           # Paulo
git checkout -b feature/arlette/densenet-finetune       # Arlette
```

### 3. Antes de entrenar — obtener el split compartido

> ⚠️ **MUY IMPORTANTE**: Todos los miembros deben usar el mismo `data/splits/splits.json`.  
> Si el archivo no está en el repo, pedírselo al miembro que lo generó primero.

```bash
# Verificar que splits.json existe
ls data/splits/splits.json

# Si NO existe, generarlo (solo el primer miembro en entrenar)
python train.py --model resnet50 --strategy straightforward \
    --member antonio --data_dir data/raw/PlantVillage
# Luego commitear data/splits/splits.json
git add data/splits/splits.json
git commit -m "feat: add shared dataset splits (seed=42, 70/15/15)"
git push origin feature/antonio/resnet-warmup
# Notificar al equipo para que hagan pull
```

### 4. Entrenar el experimento

```bash
# Ejemplo: Alondra entrena ResNet50 Straightforward
uv run python train.py \
    --model resnet50 \
    --strategy straightforward \
    --member alondra

# Ejemplo: Antonio entrena ResNet50 Warm-up
uv run python train.py \
    --model resnet50 \
    --strategy warmup \
    --member antonio

# Monitorear en TensorBoard (terminal separada)
uv run tensorboard --logdir experiments/
```

### 5. Verificar outputs generados

```bash
ls experiments/alondra/resnet50_straightforward/
# Debe contener:
# history.json, metrics.json, classification_report.txt,
# model_summary.txt, figures/, *_epoch_log.csv
```

### 6. Correr tests

```bash
uv run pytest tests/ -v
# Todos deben pasar antes del PR
```

### 7. Commit y Push

```bash
# Incluir solo los outputs necesarios (modelos .keras están en .gitignore)
git add src/ configs/
git add experiments/alondra/resnet50_straightforward/metrics.json
git add experiments/alondra/resnet50_straightforward/classification_report.txt
git add experiments/alondra/resnet50_straightforward/model_summary.txt
git add experiments/alondra/resnet50_straightforward/figures/
git add experiments/all_results.csv

git commit -m "feat(alondra): resnet50 straightforward — recall=0.9541, f1=0.9388"
git push origin feature/alondra/resnet-straightforward
```

### 8. Abrir Pull Request

1. Ir a GitHub → New Pull Request
2. Base: `develop` ← Compare: `feature/<nombre>/<experimento>`
3. Completar el template de PR (checklist incluido)
4. Solicitar review de otro miembro del equipo
5. Esperar aprobación antes de hacer merge

---

## Convenciones de Commits

Usar [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(<miembro>): <descripción breve>
fix(<miembro>): <descripción>
docs: <descripción>
test: <descripción>
refactor: <descripción>
```

**Ejemplos:**
```
feat(alondra): resnet50 straightforward baseline — recall=0.9541
feat(alondra): vgg16 straightforward baseline
feat(alondra): vgg16 finetune — recall=0.9623, f1=0.9501
feat(antonio): resnet50 warmup training — recall=0.9712
feat(antonio): densenet121 warmup — recall=0.9688
feat(paulo): resnet50 finetune — recall=0.9634
feat(paulo): densenet121 straightforward baseline
feat(arlette): densenet121 finetune — recall=0.9701
feat(arlette): vgg16 warmup — recall=0.9580
fix(arlette): correct preprocess_fn for vgg16
docs: update branching strategy in CONTRIBUTING
test: add no data leakage assertion to test_data
```

---

## Reglas de Comparabilidad

Para que los resultados del equipo sean comparables:

| Parámetro | Valor fijo | Verificar en |
|---|---|---|
| Seed | 42 | `base_config.yaml` |
| Split | 70/15/15 | `base_config.yaml` |
| Split file | `data/splits/splits.json` | Compartido via repo |
| Image size | 224×224 | `base_config.yaml` |
| Preprocesamiento | `preprocess_input` del modelo | `src/data/dataset.py` |
| Arquitectura cabeza | GAP→Dense(128)→Dropout(0.4)→Dense(1) | `src/models/base_model.py` |
| Loss | binary_crossentropy | `src/models/base_model.py` |
| Class weights | balanced | `src/training/class_weights.py` |
| EarlyStopping monitor | val_recall | `src/training/callbacks.py` |

---

## Resolución de Conflictos

Los conflictos más comunes y cómo resolverlos:

### Conflicto en `experiments/all_results.csv`
```bash
# Este archivo es append-only — resolver manualmente manteniendo TODAS las filas
git checkout --theirs experiments/all_results.csv
# Luego agregar manualmente las filas del otro branch
```

### Alguien modificó `base_config.yaml`
```bash
# NUNCA hacer merge automático de base_config.yaml
# Discutir con el equipo antes de resolver
# Cualquier cambio en base_config invalida comparabilidad
```

### Conflicto en código fuente
```bash
# Resolver manualmente, luego correr tests
pytest tests/ -v
# Si pasan todos los tests, el merge es seguro
```
