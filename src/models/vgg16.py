"""VGG16 — factory y preprocessing."""
from typing import Tuple

from tensorflow import keras

from src.models.base_model import build_classification_head
from src.utils.logger import get_logger

logger = get_logger(__name__)

BACKBONE_NAME = "vgg16"
PREPROCESS_FN = keras.applications.vgg16.preprocess_input


def build_vgg16(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    weights: str = "imagenet",
) -> keras.Model:
    """
    Construye VGG16 con cabeza de clasificación binaria.

    Nota: VGG16 tiene ~138M parámetros. El warm-up progresivo
    es especialmente importante para este backbone.

    Returns:
        Modelo con backbone nombrado 'vgg16'
    """
    base = keras.applications.VGG16(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        name=BACKBONE_NAME,
    )
    model = build_classification_head(base, dense_units, dropout_rate)
    logger.info(
        f"VGG16 construido | Parámetros totales: {model.count_params():,}"
    )
    return model
