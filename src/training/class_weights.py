"""Cálculo de class weights para manejo de desbalance de clases."""
from typing import Dict, List

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_class_weights(labels: List[int]) -> Dict[int, float]:
    """
    Calcula class weights balanceados para compensar el desbalance.

    Fórmula (sklearn 'balanced'):
        w_j = n_samples / (n_classes * n_samples_j)

    Con desbalance moderado (ej: 70% diseased / 30% healthy), esto asegura
    que la red no ignore la clase minority (healthy).

    Args:
        labels: Lista de labels del conjunto de entrenamiento

    Returns:
        Dict {clase: peso} para pasar a model.fit(class_weight=...)

    Example:
        weights = get_class_weights(y_train)
        # {0: 1.67, 1: 0.71}  → healthy tiene más peso si es minority
        model.fit(..., class_weight=weights)
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
    logger.info(f"Class weights calculados: {class_weight_dict}")
    return class_weight_dict
