"""
Pipeline de datos con tf.data para Plant Village.

Convenciones:
    - Label 0 = healthy
    - Label 1 = diseased
    - Data augmentation SOLO en train
    - Validation y test SIN augmentation
    - Usar preprocess_input del modelo (NO rescale=1./255)
"""
from pathlib import Path
from typing import Callable, List, Tuple

import tensorflow as tf

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Extensiones de imagen soportadas
IMAGE_EXTENSIONS = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]


def scan_plant_village(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Escanea un directorio estilo PlantVillage y asigna labels binarios.

    Estructura esperada:
        data_dir/
            Apple___Apple_scab/         → diseased (1)
            Apple___healthy/            → healthy  (0)
            Corn___Cercospora_leaf_spot/ → diseased (1)
            ...

    La asignación es: si "healthy" aparece en el nombre de la carpeta → 0, si no → 1

    Args:
        data_dir: Ruta al directorio raíz de PlantVillage

    Returns:
        (image_paths, binary_labels) — listas paralelas
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {data_dir}")

    image_paths: List[str] = []
    labels: List[int] = []

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir():
            continue
        label = 0 if "healthy" in class_dir.name.lower() else 1
        for ext in IMAGE_EXTENSIONS:
            for img_path in class_dir.glob(ext):
                image_paths.append(str(img_path))
                labels.append(label)

    n_healthy = labels.count(0)
    n_diseased = labels.count(1)
    logger.info(
        f"Dataset escaneado: {len(image_paths)} imágenes | "
        f"healthy={n_healthy} | diseased={n_diseased} | "
        f"ratio={n_diseased/max(1,n_healthy):.2f}x"
    )
    return image_paths, labels


def build_augmentation_layer() -> tf.keras.Sequential:
    """
    Capa de augmentation para entrenamiento.

    ⚠️ Aplicar SOLO al conjunto de train. Val y test sin augmentation.
    Los parámetros están definidos en base_config.yaml (augmentation.train).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.10),      # ±10%
            tf.keras.layers.RandomZoom(0.15),           # ±15%
            tf.keras.layers.RandomTranslation(0.10, 0.10),
            tf.keras.layers.RandomBrightness(factor=0.2),
        ],
        name="augmentation",
    )


def _load_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int],
    preprocess_fn: Callable,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Carga, redimensiona y preprocesa una imagen."""
    raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, image_size)
    img = preprocess_fn(img)
    return img, tf.cast(label, tf.float32)


def build_dataset(
    image_paths: List[str],
    labels: List[int],
    image_size: Tuple[int, int],
    preprocess_fn: Callable,
    batch_size: int = 32,
    augment: bool = False,
    shuffle: bool = False,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Construye un pipeline tf.data optimizado.

    Args:
        image_paths: Rutas a las imágenes
        labels: Labels binarios (0/1)
        image_size: (height, width) — debe ser (224, 224) para todos los modelos
        preprocess_fn: preprocess_input del modelo (ej: resnet50.preprocess_input)
        batch_size: Tamaño de batch (default: 32)
        augment: True SOLO para train, False para val/test
        shuffle: True SOLO para train
        seed: Semilla para shuffle

    Returns:
        tf.data.Dataset optimizado con prefetch

    Example:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        train_ds = build_dataset(X_train, y_train, (224, 224), preprocess_input,
                                  augment=True, shuffle=True)
        val_ds   = build_dataset(X_val, y_val, (224, 224), preprocess_input)
        test_ds  = build_dataset(X_test, y_test, (224, 224), preprocess_input)
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda path, label: _load_image(path, label, image_size, preprocess_fn),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size, drop_remainder=False)

    if augment:
        aug_layer = build_augmentation_layer()
        ds = ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)
