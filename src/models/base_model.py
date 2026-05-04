"""
Constructor base para modelos de transfer learning.

Arquitectura de la cabeza (FIJA para todos los experimentos):
    GlobalAveragePooling2D
    → Dense(128, relu)
    → Dropout(0.4)
    → Dense(1, sigmoid)

⚠️ No modificar esta arquitectura sin consenso del equipo.
"""
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_classification_head(
    base_model: keras.Model,
    dense_units: int = 128,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Añade la cabeza de clasificación binaria a un backbone pre-entrenado.

    Arquitectura fija (base_config.yaml → head):
        GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.4) → Dense(1, sigmoid)

    Args:
        base_model: Modelo backbone (ResNet50, DenseNet121, VGG16)
        dense_units: Unidades en la capa densa (default: 128)
        dropout_rate: Tasa de dropout (default: 0.4)

    Returns:
        Modelo completo (backbone + cabeza)
    """
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = keras.layers.Dropout(dropout_rate, name="dropout_head")(x)
    output = keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    return keras.Model(inputs=base_model.input, outputs=output)


def freeze_backbone(model: keras.Model, backbone_name: str) -> None:
    """
    Congela todas las capas del backbone (solo la cabeza es entrenable).

    Usado en:
        - Straightforward (única fase)
        - Fine-tuning (fase 1)
        - Warm-up (fase 1)

    Args:
        model: Modelo completo
        backbone_name: Nombre del backbone (ej: 'resnet50')
    """
    backbone = model.get_layer(backbone_name)
    backbone.trainable = False
    n_trainable = sum(1 for layer in model.layers if layer.trainable)
    n_total = len(model.layers)
    logger.info(f"Backbone '{backbone_name}' congelado | Entrenables: {n_trainable}/{n_total}")


def unfreeze_backbone(
    model: keras.Model,
    backbone_name: str,
    from_layer: int = 0,
) -> None:
    """
    Descongela el backbone desde una capa dada.

    Args:
        model: Modelo completo
        backbone_name: Nombre del backbone (ej: 'resnet50')
        from_layer: Índice desde donde descongelar.
                    0 = descongelar todo.
                    -N = descongelar las últimas N capas.

    Example:
        unfreeze_backbone(model, "resnet50", from_layer=-30)  # últimas 30 capas
        unfreeze_backbone(model, "resnet50", from_layer=0)    # todo el backbone
    """
    backbone = model.get_layer(backbone_name)
    backbone.trainable = True

    if from_layer != 0:
        total_layers = len(backbone.layers)
        freeze_until = total_layers + from_layer if from_layer < 0 else from_layer
        freeze_until = max(0, min(freeze_until, total_layers))
        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False

    n_trainable = sum(1 for layer in model.layers if layer.trainable)
    n_total = len(model.layers)
    logger.info(
        f"Backbone '{backbone_name}' descongelado desde capa {from_layer} | "
        f"Entrenables: {n_trainable}/{n_total}"
    )


def compile_model(
    model: keras.Model,
    learning_rate: float,
    optimizer_name: str = "adam",
) -> None:
    """
    Compila el modelo con loss y métricas fijas para todos los experimentos.

    Loss: binary_crossentropy (fijo)
    Métricas: accuracy, precision, recall (F1 se calcula en evaluación)

    Args:
        model: Modelo a compilar
        learning_rate: Learning rate para el optimizador
        optimizer_name: 'adam' | 'sgd' | 'rmsprop'
    """
    optimizers_map = {
        "adam": keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd": keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=learning_rate),
    }
    optimizer = optimizers_map.get(
        optimizer_name.lower(),
        keras.optimizers.Adam(learning_rate=learning_rate),
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    logger.info(
        f"Modelo compilado | optimizer={optimizer_name} | lr={learning_rate:.2e}"
    )
