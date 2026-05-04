## Descripción del PR

<!-- Explica brevemente qué hace este PR -->

**Experimento:** `<modelo>_<estrategia>` (ej: `resnet50_warmup`)  
**Responsable:** @<tu_usuario>  
**Branch:** `feature/<miembro>/<modelo>-<estrategia>`

---

## ✅ Checklist antes de solicitar review

### Código
- [ ] El código pasa `pytest tests/ -v` sin errores
- [ ] Se usó `python train.py` con los argumentos correctos
- [ ] No se modificó `base_config.yaml` sin consenso del equipo
- [ ] No se modificó `data/splits/splits.json`
- [ ] Se usó `preprocess_input` del modelo (NO `rescale=1./255`)
- [ ] Augmentation solo en train (`augment=True` solo para train_ds)

### Reproducibilidad
- [ ] El experimento usa `seed=42`
- [ ] El experimento usa `data/splits/splits.json` compartido
- [ ] La cabeza del modelo es la estándar: GAP → Dense(128) → Dropout(0.4) → Dense(1)

### Outputs (incluir en el commit)
- [ ] `experiments/<miembro>/<modelo>_<estrategia>/metrics.json`
- [ ] `experiments/<miembro>/<modelo>_<estrategia>/classification_report.txt`
- [ ] `experiments/<miembro>/<modelo>_<estrategia>/model_summary.txt`
- [ ] `experiments/<miembro>/<modelo>_<estrategia>/figures/*.png`
- [ ] Row agregado a `experiments/all_results.csv` (generado automáticamente)

---

## 📊 Resultados Obtenidos

| Métrica | Valor |
|---|---|
| **Recall** ← Prioridad | |
| Precision | |
| F1-Score | |
| Accuracy | |

**Falsos Negativos (plantas enfermas no detectadas):** 

---

## 💬 Notas / Observaciones

<!-- ¿Algo inusual? ¿Cambios de hiperparámetros justificados? ¿Issues encontrados? -->

---

## 🔗 Issues relacionados

Closes #
