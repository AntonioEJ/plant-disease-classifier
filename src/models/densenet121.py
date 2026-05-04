"""DenseNet121 — factory y preprocessing."""
from typing import Tuple

from tensorflow import keras

from src.models.base_model import build_classification_head
from src.utils.logger import get_logger

logger = get_logger(__name__)

BACKBONE_NAME = "densenet121"
PREPROCESS_FN = keras.applications.densenet.preprocess_input


def build_densenet121(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    weights: str = "imagenet",
) -> keras.Model:
    """
    Construye DenseNet121 con cabeza de clasificación binaria.

    Returns:
        Modelo con backbone nombrado 'densenet121'
    """
    base = keras.applications.DenseNet121(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        name=BACKBONE_NAME,
    )
    model = build_classification_head(base, dense_units, dropout_rate)
    logger.info(
        f"DenseNet121 construido | Parámetros totales: {model.count_params():,}"
    )
    return model
