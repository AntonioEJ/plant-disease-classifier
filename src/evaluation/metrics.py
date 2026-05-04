"""
Pipeline de evaluación estándar.

⚠️ Todos los miembros del equipo deben usar ESTE módulo para evaluar.
Garantiza que las métricas sean comparables entre experimentos.

Métricas (en orden de prioridad de negocio):
    1. Recall    ← PRIORIDAD (minimizar falsos negativos)
    2. Precision
    3. F1-score
    4. Accuracy
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["healthy", "diseased"]
RESULTS_CSV_COLUMNS = [
    "experiment", "member", "model", "strategy",
    "recall", "precision", "f1_score", "accuracy",
    "threshold", "n_test",
]


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    experiment_dir: str,
    experiment_name: str,
    member: str = "unknown",
    model_name: str = "unknown",
    strategy: str = "unknown",
    threshold: float = 0.5,
    master_csv: Optional[str] = "experiments/all_results.csv",
) -> Dict[str, Any]:
    """
    Evaluación completa de un modelo en el conjunto de test.

    Genera y guarda:
        - confusion_matrix.npy
        - classification_report.txt
        - model_summary.txt
        - metrics.json
        - Agrega fila a all_results.csv (comparación del equipo)

    Args:
        model: Modelo entrenado
        test_ds: Dataset de test (SIN augmentation, SIN shuffle)
        experiment_dir: Directorio de salida del experimento
        experiment_name: Nombre identificador (ej: 'resnet50_warmup')
        member: Miembro del equipo (antonio, alondra, paulo, arlette)
        model_name: Backbone (resnet50, densenet121, vgg16)
        strategy: Estrategia (straightforward, finetune, warmup)
        threshold: Umbral de clasificación (default: 0.5)
        master_csv: Ruta al CSV maestro de resultados del equipo

    Returns:
        Dict con todas las métricas
    """
    out_dir = Path(experiment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Evaluando {experiment_name}...")

    # Obtener predicciones
    y_prob = model.predict(test_ds, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    # Obtener labels verdaderos
    y_true = np.concatenate([y.numpy() for _, y in test_ds]).astype(int)

    # Calcular métricas
    metrics: Dict[str, Any] = {
        "experiment": experiment_name,
        "member": member,
        "model": model_name,
        "strategy": strategy,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": threshold,
        "n_test": int(len(y_true)),
        "n_healthy_test": int((y_true == 0).sum()),
        "n_diseased_test": int((y_true == 1).sum()),
    }

    # Log resumen
    logger.info(
        f"\n{'='*50}\n"
        f"  Experimento : {experiment_name}\n"
        f"  Recall      : {metrics['recall']:.4f}  ← PRIORIDAD\n"
        f"  Precision   : {metrics['precision']:.4f}\n"
        f"  F1-Score    : {metrics['f1_score']:.4f}\n"
        f"  Accuracy    : {metrics['accuracy']:.4f}\n"
        f"{'='*50}"
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    np.save(out_dir / "confusion_matrix.npy", cm)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    logger.info(f"Confusion matrix — TP={tp} | FP={fp} | FN={fn} | TN={tn}")
    logger.info(f"Falsos Negativos (plantas enfermas no detectadas): {fn}")

    # Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # Model summary
    summary_lines: list = []
    model.summary(print_fn=lambda x: summary_lines.append(x))
    (out_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Guardar métricas en JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Agregar al CSV maestro del equipo
    if master_csv:
        _append_to_master_csv(metrics, master_csv)

    return metrics


def _append_to_master_csv(metrics: Dict[str, Any], csv_path: str) -> None:
    """Agrega los resultados de un experimento al CSV maestro del equipo."""
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_file.exists()

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        row = {k: metrics.get(k, "") for k in RESULTS_CSV_COLUMNS}
        writer.writerow(row)

    logger.info(f"Resultados agregados a {csv_path}")
