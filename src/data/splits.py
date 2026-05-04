"""
División estratificada del dataset.

⚠️ CRÍTICO PARA EL EQUIPO:
    - Ejecutar create_stratified_splits UNA SOLA VEZ y compartir splits.json
    - Todos los miembros deben usar el MISMO splits.json
    - El seed=42 NO debe cambiarse
    - Si se regeneran los splits, todos los experimentos previos dejan de ser comparables
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


def create_stratified_splits(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Tuple[List, List, List, List, List, List]:
    """
    Crea splits estratificados reproducibles train/val/test.

    El split se realiza en dos pasos para mantener proporciones exactas:
        1. train vs (val + test)
        2. val vs test

    Args:
        image_paths: Lista de rutas absolutas a imágenes
        labels: Labels binarios correspondientes (0=healthy, 1=diseased)
        train_ratio: Proporción de entrenamiento (default: 0.70)
        val_ratio: Proporción de validación (default: 0.15)
        test_ratio: Proporción de test (default: 0.15)
        seed: Semilla para reproducibilidad (default: 42, NO cambiar)
        save_dir: Si se especifica, guarda los splits en splits.json

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Los ratios deben sumar exactamente 1.0"
    )

    set_global_seed(seed)

    test_val_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=test_val_ratio,
        stratify=labels,
        random_state=seed,
    )

    val_from_temp = val_ratio / test_val_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_from_temp),
        stratify=y_temp,
        random_state=seed,
    )

    _log_split_stats("train", y_train)
    _log_split_stats("val", y_val)
    _log_split_stats("test", y_test)
    logger.info(
        f"Split total — train: {len(X_train)} | val: {len(X_val)} | test: {len(X_test)}"
    )

    if save_dir:
        _save_splits(save_dir, X_train, X_val, X_test, y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_splits(splits_path: str) -> Tuple[List, List, List, List, List, List]:
    """
    Carga splits pre-computados desde JSON.

    Usar SIEMPRE este método para garantizar que todo el equipo
    usa exactamente el mismo split.

    Args:
        splits_path: Ruta al archivo splits.json

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    with open(splits_path, "r") as f:
        splits = json.load(f)

    logger.info(f"Splits cargados desde {splits_path}")
    _log_split_stats("train", splits["train"]["labels"])
    _log_split_stats("val", splits["val"]["labels"])
    _log_split_stats("test", splits["test"]["labels"])

    return (
        splits["train"]["paths"],
        splits["val"]["paths"],
        splits["test"]["paths"],
        splits["train"]["labels"],
        splits["val"]["labels"],
        splits["test"]["labels"],
    )


def _save_splits(
    save_dir: str,
    X_train: List,
    X_val: List,
    X_test: List,
    y_train: List,
    y_val: List,
    y_test: List,
) -> None:
    """Guarda los splits en JSON para compartir con el equipo."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    splits = {
        "train": {"paths": [str(p) for p in X_train], "labels": list(y_train)},
        "val": {"paths": [str(p) for p in X_val], "labels": list(y_val)},
        "test": {"paths": [str(p) for p in X_test], "labels": list(y_test)},
    }
    out_path = Path(save_dir) / "splits.json"
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    logger.info(f"Splits guardados en {out_path} — compartir con el equipo!")


def _log_split_stats(name: str, labels: List[int]) -> None:
    arr = np.array(labels)
    n_healthy = int((arr == 0).sum())
    n_diseased = int((arr == 1).sum())
    total = len(arr)
    logger.info(
        f"  {name:6s}: {total:5d} imgs | "
        f"healthy={n_healthy} ({100*n_healthy/total:.1f}%) | "
        f"diseased={n_diseased} ({100*n_diseased/total:.1f}%)"
    )
