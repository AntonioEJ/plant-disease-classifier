"""
Trainer unificado para todas las estrategias de entrenamiento.

Soporta:
    - straightforward: una sola fase con backbone congelado
    - finetune: dos fases (freeze cabeza → unfreeze parcial)
    - warmup: tres fases con descongelamiento progresivo

El trainer lee las fases directamente del config YAML, por lo que
agregar una nueva estrategia solo requiere crear un nuevo YAML.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf
from tensorflow import keras

from src.models.base_model import compile_model, freeze_backbone, unfreeze_backbone
from src.training.callbacks import build_callbacks
from src.training.class_weights import get_class_weights
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)


class Trainer:
    """
    Ejecuta el entrenamiento multi-fase de cualquier estrategia.

    Usage:
        config = load_experiment_config("resnet50", "warmup")
        trainer = Trainer(config, "experiments/antonio/resnet50_warmup")
        history = trainer.train(model, train_ds, val_ds, y_train, "resnet50")
    """

    def __init__(self, config: Dict[str, Any], experiment_dir: str) -> None:
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.strategy_cfg = config["strategy"]
        self.training_cfg = config["training"]
        self.model_cfg = config["model"]

    def train(
        self,
        model: keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        train_labels: List[int],
        backbone_name: str,
    ) -> Dict[str, List[float]]:
        """
        Ejecuta todas las fases del entrenamiento.

        Args:
            model: Modelo Keras construido (sin compilar)
            train_ds: Dataset de entrenamiento
            val_ds: Dataset de validación
            train_labels: Labels de entrenamiento (para class weights)
            backbone_name: Nombre del backbone en el modelo (ej: 'resnet50')

        Returns:
            History combinado de todas las fases
        """
        seed = self.config.get("project", {}).get("seed", 42)
        set_global_seed(seed)

        # Calcular class weights una sola vez
        class_weights = None
        if self.training_cfg.get("use_class_weights", True):
            class_weights = get_class_weights(train_labels)

        combined_history: Dict[str, List[float]] = {}
        phases = self.strategy_cfg["phases"]

        logger.info(
            f"Iniciando entrenamiento | "
            f"modelo={self.model_cfg['backbone']} | "
            f"estrategia={self.strategy_cfg['name']} | "
            f"fases={len(phases)}"
        )

        for phase_idx, phase in enumerate(phases):
            phase_name = phase["name"]
            logger.info(
                f"\n{'='*50}\n"
                f"[Fase {phase_idx + 1}/{len(phases)}] {phase_name}\n"
                f"{'='*50}"
            )

            # Configurar freeze/unfreeze del backbone
            if phase.get("freeze_backbone", True):
                freeze_backbone(model, backbone_name)
            else:
                from_layer = phase.get("unfreeze_from_layer", 0) or 0
                unfreeze_backbone(model, backbone_name, from_layer=from_layer)

            # Compilar para esta fase (posiblemente nuevo LR/optimizador)
            opt_cfg = phase["optimizer"]
            compile_model(
                model,
                learning_rate=opt_cfg["learning_rate"],
                optimizer_name=opt_cfg.get("name", "adam"),
            )

            # Callbacks con nombre por fase
            exp_name = f"{self.model_cfg['backbone']}_{self.strategy_cfg['name']}_p{phase_idx + 1}"
            callbacks = build_callbacks(
                checkpoint_dir=str(self.experiment_dir),
                experiment_name=exp_name,
                early_stopping_patience=self.training_cfg.get("early_stopping_patience", 10),
                reduce_lr_patience=self.training_cfg.get("reduce_lr_patience", 5),
                reduce_lr_factor=self.training_cfg.get("reduce_lr_factor", 0.5),
                min_lr=self.training_cfg.get("min_lr", 1e-7),
            )

            # Entrenamiento de esta fase
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=phase["epochs"],
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1,
            )

            # Acumular history de todas las fases
            for key, values in history.history.items():
                combined_history.setdefault(key, []).extend(values)

            logger.info(
                f"Fase {phase_name} completada | "
                f"Best val_recall: {max(history.history.get('val_recall', [0])):.4f}"
            )

        # Guardar history completo
        history_path = self.experiment_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(combined_history, f, indent=2)
        logger.info(f"History guardado en {history_path}")

        return combined_history
