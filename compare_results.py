"""
Agrega y compara resultados de todos los experimentos del equipo.

Uso:
    python compare_results.py
    python compare_results.py --experiments_dir experiments/

Genera:
    - Tabla en consola ordenada por Recall
    - reports/results/all_results_summary.csv
    - reports/figures/model_comparison.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from src.evaluation.visualize import plot_model_comparison
from src.utils.logger import get_logger

logger = get_logger(__name__)


def collect_all_results(experiments_dir: str = "experiments") -> pd.DataFrame:
    """
    Recorre el árbol de experiments/ y recolecta todos los metrics.json.

    Estructura esperada:
        experiments/<member>/<model>_<strategy>/metrics.json
    """
    records = []
    for metrics_file in sorted(Path(experiments_dir).rglob("metrics.json")):
        with open(metrics_file) as f:
            data = json.load(f)
        records.append(data)

    if not records:
        logger.warning(f"No se encontraron resultados en {experiments_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    cols = ["experiment", "member", "model", "strategy", "recall", "precision", "f1_score", "accuracy", "n_test"]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].sort_values("recall", ascending=False).reset_index(drop=True)
    return df


def print_results_table(df: pd.DataFrame) -> None:
    """Imprime la tabla de resultados formateada."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 120)

    print("\n" + "=" * 80)
    print("  RESULTADOS — Ordenados por Recall (Prioridad: Minimizar Falsos Negativos)")
    print("=" * 80)
    print(df.to_string(index=True))
    print()

    print("\n--- MEJOR ESTRATEGIA POR MODELO ---")
    print(f"{'Modelo':<15} {'Estrategia':<20} {'Responsable':<12} {'Recall':>8} {'F1':>8}")
    print("-" * 65)
    for model in ["resnet50", "densenet121", "vgg16"]:
        if "model" in df.columns:
            subset = df[df["model"] == model]
        else:
            subset = df[df["experiment"].str.startswith(model)]
        if not subset.empty:
            best = subset.iloc[0]
            strategy = best.get("strategy", best.get("experiment", ""))
            member = best.get("member", "")
            print(
                f"  {model:<13} {strategy:<20} {member:<12} "
                f"{best['recall']:>8.4f} {best['f1_score']:>8.4f}"
            )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparar resultados de experimentos")
    parser.add_argument("--experiments_dir", default="experiments")
    args = parser.parse_args()

    df = collect_all_results(args.experiments_dir)
    if df.empty:
        print("Sin resultados. Ejecuta train.py primero.")
        return

    print_results_table(df)

    # Guardar CSV resumen
    out_dir = Path("reports/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "all_results_summary.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Resumen guardado en {out_path}")

    # Generar gráfica de comparación
    master_csv = Path(args.experiments_dir) / "all_results.csv"
    if master_csv.exists():
        plot_model_comparison(str(master_csv), "reports/figures")


if __name__ == "__main__":
    main()
