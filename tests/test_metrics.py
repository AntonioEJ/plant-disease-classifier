"""Tests para métricas de evaluación."""
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score


class TestMetrics:

    def test_perfect_recall_zero_fn(self):
        """Recall=1.0 cuando no hay falsos negativos."""
        y_true = [1, 1, 1, 1, 0, 0]
        y_pred = [1, 1, 1, 1, 1, 0]  # FP=1, FN=0
        assert recall_score(y_true, y_pred) == 1.0

    def test_recall_penalizes_fn(self):
        """Recall debe bajar con falsos negativos (enfermos no detectados)."""
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 0, 0]  # FN=2
        assert recall_score(y_true, y_pred) == 0.5

    def test_recall_priority_over_precision(self):
        """
        Para nuestro problema, preferimos alta recall aunque baje precisión.
        Este test documenta el trade-off esperado.
        """
        # Modelo conservador (pocos FN, muchos FP)
        y_true = [1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 1, 1, 0, 0]  # FN=0, FP=2
        r = recall_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        assert r == 1.0, "Recall debe ser 1.0"
        assert p < 1.0, "Precisión sacrificada por FP"

    def test_f1_harmonic_mean(self):
        """F1 es la media armónica de precision y recall."""
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 1, 1]
        f1 = f1_score(y_true, y_pred)
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        expected_f1 = 2 * p * r / (p + r)
        assert abs(f1 - expected_f1) < 1e-6

    def test_all_healthy_predicted_diseased(self):
        """
        Si el modelo predice todo como diseased (clase 1),
        recall=1.0 pero precision es baja.
        Caso extremo a evitar.
        """
        y_true = [0, 1, 0, 1, 1]
        y_pred = [1, 1, 1, 1, 1]  # Todo predicho como 1
        assert recall_score(y_true, y_pred) == 1.0
        assert precision_score(y_true, y_pred) < 1.0


class TestConfigLoading:

    def test_config_merges_correctly(self):
        """La config merged debe tener claves de base, model y strategy."""
        import tempfile
        import yaml
        from pathlib import Path
        from src.utils.config import load_experiment_config

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "models").mkdir()
            (tmp_path / "strategies").mkdir()

            base = {"project": {"seed": 42}, "data": {"batch_size": 32}}
            model_cfg = {"model": {"name": "TestModel"}}
            strategy_cfg = {"strategy": {"name": "test_strategy", "phases": []}}

            with open(tmp_path / "base_config.yaml", "w") as f:
                yaml.dump(base, f)
            with open(tmp_path / "models" / "resnet50.yaml", "w") as f:
                yaml.dump(model_cfg, f)
            with open(tmp_path / "strategies" / "straightforward.yaml", "w") as f:
                yaml.dump(strategy_cfg, f)

            config = load_experiment_config("resnet50", "straightforward", str(tmp_path))
            assert config["project"]["seed"] == 42
            assert config["model"]["name"] == "TestModel"
            assert config["strategy"]["name"] == "test_strategy"
