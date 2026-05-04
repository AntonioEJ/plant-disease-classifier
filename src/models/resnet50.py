"""ResNet50 — factory y preprocessing."""
from typing import Tuple

from tensorflow import keras

from src.models.base_model import build_classification_head
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Nombre fijo del backbone — usado para freeze/unfreeze
BACKBONE_NAME = "resnet50"

# Función de preprocesamiento específica del modelo
# ⚠️ NO usar rescale=1./255 — usar esta función
PREPROCESS_FN = keras.applications.resnet50.preprocess_input


def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dense_units: int = 128,
    dropout_rate: float = 0.4,
    weights: str = "imagenet",
) -> keras.Model:
    """
    Construye ResNet50 con cabeza de clasificación binaria.

    Args:
        input_shape: Forma de entrada (debe ser (224, 224, 3))
        dense_units: Unidades en la capa densa de la cabeza
        dropout_rate: Tasa de dropout
        weights: Pesos pre-entrenados ('imagenet' o None)

    Returns:
        Modelo con backbone nombrado 'resnet50' (necesario para freeze/unfreeze)
    """
    base = keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        name=BACKBONE_NAME,
    )
    model = build_classification_head(base, dense_units, dropout_rate)
    logger.info(
        f"ResNet50 construido | Parámetros totales: {model.count_params():,}"
    )
    return model
