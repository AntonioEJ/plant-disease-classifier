# Guía de Contribución — Plant Disease Classifier

## Distribución de Experimentos

| Modelo      | Estrategia      | Responsable | Rama actual   |
|-------------|-----------------|-------------|--------------|
| ResNet50    | Straightforward | Alondra     | `alondra`    |
| ResNet50    | Fine-tuning     | Paulo       | `paulo`      |
| ResNet50    | Warm-up         | Antonio     | `antonio`    |
| DenseNet121 | Straightforward | Paulo       | `paulo`      |
| DenseNet121 | Fine-tuning     | Arlette     | `arlette`    |
| DenseNet121 | Warm-up         | Antonio     | `antonio`    |
| VGG16       | Straightforward | Alondra     | `alondra`    |
| VGG16       | Fine-tuning     | Alondra     | `alondra`    |
| VGG16       | Warm-up         | Arlette     | `arlette`    |

---

## 7. Subir tus resultados
# Guía de Contribución — Plant Disease Classifier

## ¿Cómo contribuir?


1. **Usa tu rama personal** (ya creada para cada miembro):
  - antonio → rama `antonio`
  - alondra → rama `alondra`
  - paulo   → rama `paulo`
  - arlette → rama `arlette`

  Si eres nuevo, crea tu rama con tu nombre:
  ```bash
  git checkout -b tu-nombre
  ```
2. Trabaja siempre en tu rama personal. Haz todos tus experimentos y notebooks ahí.
3. Crea tu carpeta en `experiments/` con tu nombre y modelo/estrategia.
4. Usa los notebooks de `notebooks/` para hacer splits, cargar datos, entrenar y analizar resultados.
5. Guarda tus resultados (métricas, gráficos, notas) en tu carpeta de `experiments/`.
6. Si generas nuevos splits, compártelos en `data/splits/` o por Drive.
7. No subas datos originales ni archivos pesados al repositorio.


## Buenas prácticas

- Trabaja siempre desde tu rama personal y desde notebooks, no scripts.
- Documenta tus experimentos y pasos en los notebooks.
- Si tienes dudas, pregunta en el grupo.
- No modifiques ni borres archivos de otros compañeros.

## Estructura recomendada para tu experimento

```
experiments/
  └── tu-nombre/
      └── modelo_estrategia/
          ├── notebook_experimento.ipynb
          ├── metrics.json
          ├── classification_report.txt
          ├── model_summary.txt
          └── figures/
```


## ¿Cómo compartir resultados?

1. Sube solo archivos ligeros y relevantes (métricas, reportes, figuras) a tu rama personal.
2. Usa Drive o enlaces externos para archivos pesados o datos.
3. Si haces cambios importantes, avisa al grupo.
4. Cuando quieras compartir resultados finales, haz un Pull Request de tu rama personal a `develop` o `main`.

---

¡Gracias por contribuir! Todo el flujo es notebook-only y colaborativo.

**Tips:**
- Si tienes dudas, pregunta en el grupo.
- No modifiques archivos de otros compañeros.
- Si hay conflicto, pregunta antes de resolver.
- Usa mensajes de commit claros: `feat(tu-nombre): modelo estrategia — métrica`

---
docs: <descripción>
