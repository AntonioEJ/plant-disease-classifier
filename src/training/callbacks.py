"""Stack de callbacks estándar para todos los experimentos."""
from pathlib import Path
from typing import List

from tensorflow import keras

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_callbacks(
    checkpoint_dir: str,
    experiment_name: str,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-7,
    monitor: str = "val_recall",
) -> List[keras.callbacks.Callback]:
    """
    Stack de callbacks estándar para todos los experimentos.

    Callbacks incluidos:
        1. ModelCheckpoint  → guarda el mejor modelo por val_recall
        2. EarlyStopping    → detiene si val_recall no mejora
        3. ReduceLROnPlateau → reduce LR cuando val_loss se estanca
        4. TensorBoard      → logging para visualización
        5. CSVLogger        → registro de métricas por época

    ⚠️ El monitor es val_recall (prioridad de negocio: minimizar FN).
       NO cambiar a val_accuracy.

    Args:
        checkpoint_dir: Directorio para guardar checkpoints y logs
        experiment_name: Prefijo para nombres de archivos
        early_stopping_patience: Épocas sin mejora antes de detener
        reduce_lr_patience: Épocas sin mejora antes de reducir LR
        reduce_lr_factor: Factor de reducción del LR
        min_lr: LR mínimo (límite inferior)
        monitor: Métrica a monitorear (default: val_recall)

    Returns:
        Lista de callbacks Keras configurados
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        # 1. Guardar mejor modelo según val_recall
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path / f"{experiment_name}_best.keras"),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # 2. Detener si no hay mejora en val_recall
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # 3. Reducir LR cuando val_loss se estanca
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        # 4. TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=str(ckpt_path / "tensorboard"),
            histogram_freq=0,
            write_graph=False,
            update_freq="epoch",
        ),
        # 5. CSV de métricas por época
        keras.callbacks.CSVLogger(
            filename=str(ckpt_path / f"{experiment_name}_epoch_log.csv"),
            append=False,
        ),
    ]

    logger.info(
        f"Callbacks configurados | monitor={monitor} | "
        f"patience={early_stopping_patience} | dir={checkpoint_dir}"
    )
    return callbacks
