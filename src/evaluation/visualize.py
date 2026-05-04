"""Visualizaciones para history de entrenamiento y resultados de evaluación."""
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Backend no-interactivo (compatible con servidores y Colab)

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["healthy", "diseased"]
PALETTE = {"train": "#2196F3", "val": "#FF5722"}


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: str,
    experiment_name: str,
) -> None:
    """
    Grafica las curvas de entrenamiento para todas las métricas.

    Genera: <experiment_name>_history.png

    Args:
        history: Dict del history combinado de todas las fases
        save_dir: Directorio de salida
        experiment_name: Nombre del experimento (usado en título y nombre de archivo)
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    metrics_to_plot = [
        ("loss", "Loss"),
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall ← Prioridad"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training History — {experiment_name}", fontsize=14, fontweight="bold")

    for ax, (metric_key, label) in zip(axes.flatten(), metrics_to_plot):
        train_key = metric_key
        val_key = f"val_{metric_key}"

        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], color=PALETTE["train"],
                    label="Train", linewidth=2)
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], color=PALETTE["val"],
                    label="Validation", linewidth=2, linestyle="--")

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Época")
        ax.set_ylabel(label.split(" ")[0])
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig_path = save_path / f"{experiment_name}_history.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"History plot guardado: {fig_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    save_dir: str,
    experiment_name: str,
) -> None:
    """
    Grafica la matriz de confusión como heatmap.

    Genera: <experiment_name>_confusion_matrix.png

    Args:
        cm: numpy array (2x2) de la confusion matrix
        save_dir: Directorio de salida
        experiment_name: Nombre del experimento
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_ylabel("Etiqueta real", fontsize=12)
    ax.set_xlabel("Etiqueta predicha", fontsize=12)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=12, fontweight="bold")

    # Destacar celda de Falsos Negativos (importante para el negocio)
    if cm.shape == (2, 2):
        fn = cm[1, 0]  # Diseased predicho como Healthy
        ax.set_xlabel(
            f"Etiqueta predicha\n⚠️ FN={fn} (plantas enfermas no detectadas)",
            fontsize=11,
        )

    plt.tight_layout()
    fig_path = save_path / f"{experiment_name}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix guardada: {fig_path}")


def plot_model_comparison(
    results_csv: str,
    save_dir: str,
) -> None:
    """
    Genera gráficas de comparación de todos los experimentos del equipo.

    Args:
        results_csv: Ruta al CSV maestro de resultados (all_results.csv)
        save_dir: Directorio de salida
    """
    import pandas as pd

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)
    if df.empty:
        logger.warning("No hay resultados para comparar.")
        return

    metrics = ["recall", "precision", "f1_score", "accuracy"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Comparación de Experimentos — Plant Disease Classifier", fontsize=14)

    for ax, metric in zip(axes.flatten(), metrics):
        df_sorted = df.sort_values(metric, ascending=True)
        colors = ["#4CAF50" if metric == "recall" else "#2196F3"] * len(df_sorted)
        ax.barh(df_sorted["experiment"], df_sorted[metric], color=colors, alpha=0.8)
        ax.set_title(f"{metric.upper()}" + (" ← PRIORIDAD" if metric == "recall" else ""))
        ax.set_xlim(0, 1)
        ax.axvline(x=0.9, color="red", linestyle="--", alpha=0.5, label="Target: 0.90")
        ax.legend(fontsize=9)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig_path = save_path / "model_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparación guardada: {fig_path}")
