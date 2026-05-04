"""Utilidades para reproducibilidad. Llamar SIEMPRE antes de cualquier experimento."""
import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """
    Fija todas las semillas aleatorias para reproducibilidad total.

    Debe llamarse antes de:
        - importar/construir modelos
        - crear datasets
        - iniciar entrenamiento

    Args:
        seed: Semilla global. Por defecto 42 para todos los experimentos.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
