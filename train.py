"""
Script principal de entrenamiento.

Uso:
    python train.py --model resnet50 --strategy warmup --member antonio

Esto:
    1. Carga la config para el modelo + estrategia
    2. Carga los splits compartidos (o los genera si no existen)
    3. Construye y entrena el modelo
    4. Guarda todos los outputs en experiments/<member>/<model>_<strategy>/

Ejemplos completos:
    # Antonio — ResNet50 Straightforward
    python train.py --model resnet50 --strategy straightforward --member antonio

    # Paulo — DenseNet121 Straightforward
    python train.py --model densenet121 --strategy straightforward --member paulo

    # Arlette — VGG16 Warm-up
    python train.py --model vgg16 --strategy warmup --member arlette
"""
import argparse
import sys
from pathlib import Path

# Asegurar que src/ es importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.data.dataset import build_dataset, scan_plant_village
from src.data.splits import create_stratified_splits, load_splits
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import plot_confusion_matrix, plot_training_history
from src.training.trainer import Trainer
from src.utils.config import load_experiment_config
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed

logger = get_logger(__name__)

# Mapa de factories de modelos
_MODEL_REGISTRY = {
    "resnet50": ("src.models.resnet50", "build_resnet50", "PREPROCESS_FN"),
    "densenet121": ("src.models.densenet121", "build_densenet121", "PREPROCESS_FN"),
    "vgg16": ("src.models.vgg16", "build_vgg16", "PREPROCESS_FN"),
}


def _load_model_factory(model_name: str):
    """Importa lazily el builder y preprocess_fn del modelo."""
    module_path, builder_name, preprocess_name = _MODEL_REGISTRY[model_name]
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, builder_name), getattr(module, preprocess_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenar un clasificador de enfermedades en plantas.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(_MODEL_REGISTRY.keys()),
        help="Arquitectura del backbone",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["straightforward", "finetune", "warmup"],
        help="Estrategia de entrenamiento",
    )
    parser.add_argument(
        "--member",
        required=True,
        choices=["antonio", "alondra", "paulo", "arlette"],
        help="Miembro del equipo responsable del experimento",
    )
    parser.add_argument(
        "--data_dir",
        default="data/raw/PlantVillage",
        help="Ruta al dataset PlantVillage (default: data/raw/PlantVillage)",
    )
    parser.add_argument(
        "--splits_file",
        default="data/splits/splits.json",
        help="Ruta al JSON de splits pre-computados",
    )
    parser.add_argument(
        "--configs_dir",
        default="configs",
        help="Ruta al directorio de configs (default: configs/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Cargar config completa
    config = load_experiment_config(args.model, args.strategy, args.configs_dir)

    seed = config.get("project", {}).get("seed", 42)
    set_global_seed(seed)

    experiment_name = f"{args.model}_{args.strategy}"
    experiment_dir = Path("experiments") / args.member / experiment_name
    image_size = tuple(config["data"]["image_size"])

    logger.info("=" * 60)
    logger.info(f"  Experimento : {experiment_name}")
    logger.info(f"  Miembro     : {args.member}")
    logger.info(f"  Seed        : {seed}")
    logger.info(f"  Output dir  : {experiment_dir}")
    logger.info("=" * 60)

    # Construir modelo
    build_fn, preprocess_fn = _load_model_factory(args.model)
    model = build_fn(
        input_shape=(*image_size, 3),
        dense_units=config["head"]["dense_units"],
        dropout_rate=config["head"]["dropout_rate"],
    )

    # Cargar o crear splits
    splits_file = Path(args.splits_file)
    if splits_file.exists():
        logger.info(f"Usando splits pre-computados: {splits_file}")
        X_train, X_val, X_test, y_train, y_val, y_test = load_splits(str(splits_file))
    else:
        logger.info("Splits no encontrados — generando y guardando...")
        logger.warning(
            "⚠️ Comparte data/splits/splits.json con el equipo "
            "para garantizar comparabilidad!"
        )
        image_paths, labels = scan_plant_village(args.data_dir)
        data_cfg = config["data"]
        X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(
            image_paths,
            labels,
            train_ratio=data_cfg["train_ratio"],
            val_ratio=data_cfg["val_ratio"],
            test_ratio=data_cfg["test_ratio"],
            seed=seed,
            save_dir="data/splits",
        )

    # Construir pipelines tf.data
    batch_size = config["data"]["batch_size"]
    train_ds = build_dataset(
        X_train, y_train, image_size, preprocess_fn,
        batch_size=batch_size, augment=True, shuffle=True, seed=seed,
    )
    val_ds = build_dataset(
        X_val, y_val, image_size, preprocess_fn,
        batch_size=batch_size, augment=False, shuffle=False,
    )
    test_ds = build_dataset(
        X_test, y_test, image_size, preprocess_fn,
        batch_size=batch_size, augment=False, shuffle=False,
    )

    # Entrenar
    trainer = Trainer(config, str(experiment_dir))
    history = trainer.train(model, train_ds, val_ds, y_train, backbone_name=args.model)

    # Evaluar
    metrics = evaluate_model(
        model=model,
        test_ds=test_ds,
        experiment_dir=str(experiment_dir),
        experiment_name=experiment_name,
        member=args.member,
        model_name=args.model,
        strategy=args.strategy,
    )

    # Visualizaciones
    figures_dir = str(experiment_dir / "figures")
    plot_training_history(history, figures_dir, experiment_name)
    cm = np.load(experiment_dir / "confusion_matrix.npy")
    plot_confusion_matrix(cm, figures_dir, experiment_name)

    # Resumen final
    logger.info("\n" + "=" * 60)
    logger.info("  ENTRENAMIENTO COMPLETADO")
    logger.info(f"  Recall    : {metrics['recall']:.4f}  ← PRIORIDAD DE NEGOCIO")
    logger.info(f"  F1-Score  : {metrics['f1_score']:.4f}")
    logger.info(f"  Precision : {metrics['precision']:.4f}")
    logger.info(f"  Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"  Outputs   : {experiment_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
