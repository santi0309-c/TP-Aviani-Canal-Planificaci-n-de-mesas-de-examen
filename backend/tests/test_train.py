"""
test_train.py — Pruebas unitarias para el entrenamiento y comparación de modelos (Ticket T09).
"""

import os
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from app.ml.features import build_features, load_dataset, split_data
from app.ml.train import (
    format_table,
    get_candidate_models,
    train,
    train_and_compare,
)

DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "pares_materias.csv")


@pytest.fixture
def dataset() -> pd.DataFrame:
    """Carga el dataset sintético."""
    return load_dataset(DATA_PATH)


@pytest.fixture
def data_splits(dataset: pd.DataFrame) -> dict:
    """Genera las particiones con semilla fija 42."""
    return split_data(dataset, seed=42)


class TestCandidateModels:
    """Verifica que los modelos candidatos estén correctamente configurados."""

    def test_three_candidate_families_present(self):
        """Deben existir las 3 familias solicitadas: Regresión Lineal, Árbol y KNN."""
        candidates = get_candidate_models()
        assert "Regresión Lineal" in candidates
        assert "Árbol de Decisión" in candidates
        assert "KNN" in candidates

    def test_candidates_have_pipelines_with_preprocessor(self):
        """Cada candidato debe ser un Pipeline que incluya el preprocesador."""
        candidates = get_candidate_models()
        for name, (pipeline, params) in candidates.items():
            assert hasattr(pipeline, "fit")
            assert hasattr(pipeline, "predict")
            assert "preprocessor" in pipeline.named_steps
            assert "regressor" in pipeline.named_steps


class TestTrainAndCompare:
    """Verifica el proceso de ajuste y evaluación de modelos."""

    def test_train_and_compare_returns_expected_structure(self, data_splits: dict):
        """Debe retornar resultados para cada modelo y designar un ganador."""
        X_train, y_train = data_splits["X_train"], data_splits["y_train"]
        X_val, y_val = data_splits["X_val"], data_splits["y_val"]

        comparison = train_and_compare(X_train, y_train, X_val, y_val, cv_folds=3)

        assert "results" in comparison
        assert "best_model_name" in comparison
        assert "best_model" in comparison

        results = comparison["results"]
        assert len(results) == 3

        for name in ["Regresión Lineal", "Árbol de Decisión", "KNN"]:
            res = results[name]
            assert "val_mae" in res
            assert "val_rmse" in res
            assert "val_r2" in res
            assert res["val_mae"] >= 0.0
            assert res["val_rmse"] >= 0.0
            assert 0.0 <= res["val_r2"] <= 1.0

    def test_best_model_matches_lowest_val_rmse(self, data_splits: dict):
        """El modelo elegido debe tener el menor RMSE sobre el conjunto de validación."""
        X_train, y_train = data_splits["X_train"], data_splits["y_train"]
        X_val, y_val = data_splits["X_val"], data_splits["y_val"]

        comparison = train_and_compare(X_train, y_train, X_val, y_val, cv_folds=3)
        results = comparison["results"]
        best_name = comparison["best_model_name"]

        best_rmse = results[best_name]["val_rmse"]
        for name, res in results.items():
            assert best_rmse <= res["val_rmse"]

    def test_format_table_contains_all_models(self, data_splits: dict):
        """La tabla de comparación debe incluir los nombres de todos los modelos."""
        X_train, y_train = data_splits["X_train"], data_splits["y_train"]
        X_val, y_val = data_splits["X_val"], data_splits["y_val"]

        comparison = train_and_compare(X_train, y_train, X_val, y_val, cv_folds=3)
        table = format_table(comparison["results"])

        assert "Regresión Lineal" in table
        assert "Árbol de Decisión" in table
        assert "KNN" in table


class TestTrainCliAndPersistence:
    """Verifica el flujo integral de entrenamiento y guardado con joblib."""

    def test_train_generates_joblib_file_and_can_predict(self):
        """Debe persistir el modelo en disco y poder cargarse para predecir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "modelo.joblib")
            comparison = train(data_path=DATA_PATH, out_path=out_path, seed=42)

            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 0

            # Cargar el modelo guardado
            loaded_model = joblib.load(out_path)
            assert hasattr(loaded_model, "predict")

            # Realizar predicción de prueba sobre un par sintético
            sample_X = pd.DataFrame([{
                "cant_estudiantes_compartidos_historico": 50,
                "mismo_anio": 1,
                "misma_carrera": 1,
            }])
            pred = loaded_model.predict(sample_X)
            assert isinstance(pred, np.ndarray)
            assert len(pred) == 1
            assert pred[0] > 0
