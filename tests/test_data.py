"""Tests para el pipeline de datos."""
import json
import tempfile
from pathlib import Path

import pytest

from src.data.splits import create_stratified_splits, load_splits


def _make_dummy_data(n_healthy: int = 50, n_diseased: int = 50):
    """Genera datos dummy para testing."""
    paths = [f"img_{i}.jpg" for i in range(n_healthy + n_diseased)]
    labels = [0] * n_healthy + [1] * n_diseased
    return paths, labels


# ──────────────────────────── SPLITS ─────────────────────────────────────────

class TestStratifiedSplits:

    def test_split_sizes_exact(self):
        """Los tamaños de split deben ser exactamente 70/15/15."""
        paths, labels = _make_dummy_data(50, 50)
        X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(
            paths, labels, 0.70, 0.15, 0.15, seed=42
        )
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15

    def test_split_ratios_sum_to_one(self):
        """Ratios que no suman 1.0 deben lanzar AssertionError."""
        paths, labels = _make_dummy_data()
        with pytest.raises(AssertionError):
            create_stratified_splits(paths, labels, 0.60, 0.20, 0.30, seed=42)

    def test_stratification_balance(self):
        """Cada split debe mantener la proporción de clases."""
        paths, labels = _make_dummy_data(100, 100)
        _, _, _, y_train, y_val, y_test = create_stratified_splits(
            paths, labels, 0.70, 0.15, 0.15, seed=42
        )
        # Tolerar ±2 muestras por variación de estratificación
        assert abs(y_train.count(0) - y_train.count(1)) <= 2
        assert abs(y_val.count(0) - y_val.count(1)) <= 2
        assert abs(y_test.count(0) - y_test.count(1)) <= 2

    def test_reproducibility_same_seed(self):
        """El mismo seed debe producir splits idénticos."""
        paths, labels = _make_dummy_data()
        split1 = create_stratified_splits(paths, labels, seed=42)
        split2 = create_stratified_splits(paths, labels, seed=42)
        assert split1[0] == split2[0], "X_train debe ser idéntico con mismo seed"
        assert split1[3] == split2[3], "y_train debe ser idéntico con mismo seed"

    def test_different_seeds_differ(self):
        """Seeds distintos deben producir splits distintos."""
        paths, labels = _make_dummy_data()
        split1 = create_stratified_splits(paths, labels, seed=42)
        split2 = create_stratified_splits(paths, labels, seed=99)
        assert split1[0] != split2[0], "Seeds distintos deben producir splits distintos"

    def test_no_data_leakage(self):
        """No debe haber imágenes duplicadas entre train/val/test."""
        paths, labels = _make_dummy_data()
        X_train, X_val, X_test, *_ = create_stratified_splits(
            paths, labels, seed=42
        )
        train_set = set(X_train)
        val_set = set(X_val)
        test_set = set(X_test)
        assert len(train_set & val_set) == 0, "Data leakage: train ∩ val no vacío"
        assert len(train_set & test_set) == 0, "Data leakage: train ∩ test no vacío"
        assert len(val_set & test_set) == 0, "Data leakage: val ∩ test no vacío"

    def test_save_and_load_splits(self):
        """Guardar y cargar splits debe producir los mismos datos."""
        paths, labels = _make_dummy_data()
        with tempfile.TemporaryDirectory() as tmp_dir:
            X_train, X_val, X_test, y_train, y_val, y_test = create_stratified_splits(
                paths, labels, seed=42, save_dir=tmp_dir
            )
            splits_file = Path(tmp_dir) / "splits.json"
            assert splits_file.exists()

            loaded = load_splits(str(splits_file))
            assert loaded[0] == X_train
            assert loaded[3] == y_train
