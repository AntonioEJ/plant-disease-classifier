"""src/evaluation package."""
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import (
    plot_training_history,
    plot_confusion_matrix,
    plot_model_comparison,
)

__all__ = [
    "evaluate_model",
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_model_comparison",
]
