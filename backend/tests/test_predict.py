"""
test_predict.py — Pruebas unitarias para la carga, persistencia y predicción del modelo (Ticket T12).
"""

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.ml.features import load_dataset
from app.ml.predict import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    clear_model_cache,
    load_model,
    predict,
    predecir_pares,
)

DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "pares_materias.csv")
MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "modelo.joblib")


@pytest.fixture(autouse=True)
def reset_cache():
    """Limpia la caché del modelo antes y después de cada test."""
    clear_model_cache()
    yield
    clear_model_cache()


class TestLoadModel:
    """Verifica la carga del modelo con persistencia y patrón Singleton / Caché."""

    def test_load_model_returns_pipeline(self):
        """load_model() debe retornar un objeto Pipeline con preprocessor y regressor."""
        model = load_model(MODEL_PATH)
        assert isinstance(model, Pipeline)
        assert "preprocessor" in model.named_steps
        assert "regressor" in model.named_steps
        assert hasattr(model, "predict")

    def test_load_model_caching(self):
        """Invocaciones consecutivas a load_model() deben retornar el mismo objeto en memoria."""
        model_1 = load_model(MODEL_PATH)
        model_2 = load_model(MODEL_PATH)

        # Mismo identificador de memoria (Singleton / Caché activo)
        assert model_1 is model_2
        assert id(model_1) == id(model_2)

    def test_load_model_reload_flag(self):
        """Si reload=True, debe forzar una nueva lectura de disco."""
        model_1 = load_model(MODEL_PATH)
        model_2 = load_model(MODEL_PATH, reload=True)

        assert model_1 is not model_2
        assert isinstance(model_2, Pipeline)

    def test_clear_model_cache(self):
        """clear_model_cache() debe vaciar la referencia en memoria."""
        model_1 = load_model(MODEL_PATH)
        clear_model_cache()
        model_2 = load_model(MODEL_PATH)

        assert model_1 is not model_2

    def test_load_model_file_not_found(self):
        """Debe lanzar FileNotFoundError si se apunta a una ruta inexistente y auto_train=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = os.path.join(tmpdir, "modelo_que_no_existe.joblib")
            with pytest.raises(FileNotFoundError):
                load_model(non_existent, auto_train=False)

    def test_load_model_auto_train_fallback(self):
        """Si el modelo no existe pero auto_train=True, debe entrenarlo y cargarlo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = os.path.join(tmpdir, "nuevo_modelo.joblib")
            assert not os.path.exists(temp_model_path)

            model = load_model(temp_model_path, auto_train=True)
            assert isinstance(model, Pipeline)
            assert os.path.exists(temp_model_path)


class TestPredecirPares:
    """Verifica la función principal de inferencia sobre pares de materias."""

    def test_predecir_pares_acceptance_criterion(self):
        """
        Criterio de aceptación T12:
        Dado el dataset institucional (40 materias), predecir_pares() debe devolver
        exactamente 780 predicciones (comb(40, 2)).
        """
        predictions = predecir_pares(data_path=DATA_PATH)
        assert isinstance(predictions, list)
        assert len(predictions) == 780

    def test_predecir_pares_structure_and_types(self):
        """Cada registro devuelto debe contener los campos requeridos por el grafo y valores >= 0."""
        predictions = predecir_pares(data_path=DATA_PATH)
        assert len(predictions) > 0

        first = predictions[0]
        # Claves esperadas por build_conflict_graph (T13) y la API (T17/T18)
        assert "materia_1" in first
        assert "materia_2" in first
        assert "conflicto_predicho" in first
        assert "cant_estudiantes_afectados" in first

        for item in predictions[:50]:  # Validar muestra representativa
            assert isinstance(item["materia_1"], str)
            assert isinstance(item["materia_2"], str)
            assert isinstance(item["conflicto_predicho"], (int, float))
            assert item["conflicto_predicho"] >= 0.0

            # No puede superar los estudiantes compartidos históricos si existen
            if "cant_estudiantes_compartidos_historico" in item:
                assert item["conflicto_predicho"] <= item["cant_estudiantes_compartidos_historico"] + 1e-5

    def test_predecir_pares_dataframe_input(self):
        """predecir_pares debe admitir un subset como pd.DataFrame directamente."""
        df_full = load_dataset(DATA_PATH)
        subset_df = df_full.head(15).copy()

        preds = predecir_pares(df=subset_df, as_dict=True)
        assert isinstance(preds, list)
        assert len(preds) == 15
        assert preds[0]["materia_1"] == str(subset_df.iloc[0]["materia_id_1"])
        assert preds[0]["materia_2"] == str(subset_df.iloc[0]["materia_id_2"])

    def test_predecir_pares_returns_dataframe(self):
        """Con as_dict=False, debe devolver un pd.DataFrame enriquecido."""
        df_full = load_dataset(DATA_PATH)
        subset_df = df_full.head(20).copy()

        result_df = predecir_pares(df=subset_df, as_dict=False)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 20
        assert "conflicto_predicho" in result_df.columns
        assert (result_df["conflicto_predicho"] >= 0.0).all()


class TestPredictGeneric:
    """Verifica la función predict() genérica sobre matrices y DataFrames."""

    def test_predict_dataframe_with_feature_columns(self):
        """Debe procesar DataFrames seleccionando automáticamente las columnas de features."""
        model = load_model(MODEL_PATH)
        sample_df = pd.DataFrame([
            {"cant_estudiantes_compartidos_historico": 100, "mismo_anio": True, "misma_carrera": True},
            {"cant_estudiantes_compartidos_historico": 0, "mismo_anio": False, "misma_carrera": False},
        ])

        y_pred = predict(model, sample_df)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == 2
        assert y_pred[0] > y_pred[1]
        assert (y_pred >= 0.0).all()

    def test_predict_list_of_dicts(self):
        """Debe aceptar una lista de diccionarios con las features."""
        model = load_model(MODEL_PATH)
        sample_list = [
            {"cant_estudiantes_compartidos_historico": 50, "mismo_anio": 1, "misma_carrera": 1},
        ]
        y_pred = predict(model, sample_list)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_pred) == 1
        assert y_pred[0] > 0.0


class TestCliExport:
    """Verifica la exportación de predicciones por CLI."""

    def test_cli_json_export(self):
        """Debe permitir exportar predicciones a un archivo JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "predicciones.json")
            preds = predecir_pares(data_path=DATA_PATH, as_dict=True)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)

            assert os.path.exists(out_file)
            with open(out_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert len(loaded) == 780
            assert "conflicto_predicho" in loaded[0]
