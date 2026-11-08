"""
test_error_analysis.py — Pruebas unitarias para el diagnóstico y análisis de errores (Ticket T11).
"""

import json
import os
from pathlib import Path
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from app.ml.error_analysis import (
    DEFAULT_DATA,
    DEFAULT_MATERIAS,
    DEFAULT_MODEL,
    build_error_dataframe,
    compute_error_statistics,
    get_border_cases,
    get_top_errors,
    run_error_analysis,
)
from app.ml.features import load_dataset, split_data


@pytest.fixture(scope="module")
def expected_test_size() -> int:
    """Calcula el tamaño esperado del conjunto de test de forma dinámica a partir de la partición."""
    df_raw = load_dataset(str(DEFAULT_DATA))
    split = split_data(df_raw, seed=42)
    return len(split["X_test"])


@pytest.fixture(scope="module")
def error_data():
    """Genera el DataFrame de diagnóstico de errores para los tests."""
    test_df, meta = build_error_dataframe(
        data_path=DEFAULT_DATA,
        model_path=DEFAULT_MODEL,
        materias_path=DEFAULT_MATERIAS,
        seed=42,
    )
    return test_df, meta


class TestErrorDataframe:
    """Verifica la correcta construcción y enriquecimiento del DataFrame de test."""

    def test_test_dataframe_shape_and_columns(self, error_data, expected_test_size: int):
        test_df, meta = error_data
        # Validación dinámica del tamaño sin hardcodear 117
        assert meta["n_test"] == expected_test_size
        assert len(test_df) == expected_test_size

        expected_cols = [
            "materia_id_1",
            "materia_id_2",
            "materia_1_nombre",
            "materia_2_nombre",
            "y_real",
            "y_pred",
            "residuo",
            "error_abs",
            "conflicto_real",
            "conflicto_pred",
            "es_falso_negativo",
            "es_falso_positivo",
        ]
        for col in expected_cols:
            assert col in test_df.columns

    def test_residual_calculation_consistency(self, error_data):
        test_df, _ = error_data
        calculated_res = np.round(test_df["y_real"] - test_df["y_pred"], 2)
        pd.testing.assert_series_equal(test_df["residuo"], calculated_res, check_names=False)

        calculated_abs = np.round(np.abs(test_df["residuo"]), 2)
        pd.testing.assert_series_equal(test_df["error_abs"], calculated_abs, check_names=False)


class TestErrorStatistics:
    """Verifica el cálculo de estadísticas de sesgo y agrupamientos."""

    def test_bias_statistics_and_balance(self, error_data, expected_test_size: int):
        test_df, _ = error_data
        stats = compute_error_statistics(test_df)
        sesgo = stats["sesgo"]

        # El estimador debe ser insesgado (media de residuos cercana a 0)
        assert abs(sesgo["media_residuos"]) < 0.5

        # Subestimados + sobreestimados + exactos deben sumar el total de pares de test dinámicamente
        total_sub_sobre = sesgo["cant_subestimados"] + sesgo["cant_sobreestimados"] + sesgo["cant_exactos"]
        assert total_sub_sobre == expected_test_size

        # Proporción equilibrada (~50% sub, ~50% sobre)
        assert 40.0 <= sesgo["pct_subestimados"] <= 60.0

    def test_group_breakdowns_counts(self, error_data, expected_test_size: int):
        test_df, _ = error_data
        stats = compute_error_statistics(test_df)

        mismo_anio = stats["desglose_mismo_anio"]
        total_anio = mismo_anio[True]["count"] + mismo_anio[False]["count"]
        assert total_anio == expected_test_size

        misma_carrera = stats["desglose_misma_carrera"]
        total_carrera = misma_carrera[True]["count"] + misma_carrera[False]["count"]
        assert total_carrera == expected_test_size


class TestTopErrorsAndBorderCases:
    """Verifica la detección de casos críticos y frontera."""

    def test_top_errors_ordering(self, error_data):
        test_df, _ = error_data
        top10 = get_top_errors(test_df, n=10)

        assert len(top10) == 10
        # Primer elemento debe tener el error absoluto máximo
        assert top10[0]["error_abs"] == test_df["error_abs"].max()

        # Orden decreciente
        for i in range(len(top10) - 1):
            assert top10[i]["error_abs"] >= top10[i + 1]["error_abs"]

    def test_border_cases_counts_and_consistency(self, error_data):
        test_df, meta = error_data
        border = get_border_cases(test_df)
        umbral = meta["umbral"]

        fn_list = border["falsos_negativos"]
        fp_list = border["falsos_positivos"]

        # Validación dinámica según las columnas booleanas calculadas en el DataFrame
        expected_fn = int(test_df["es_falso_negativo"].sum())
        expected_fp = int(test_df["es_falso_positivo"].sum())

        assert len(fn_list) == expected_fn
        assert len(fp_list) == expected_fp

        # Todos los FN deben tener y_real >= umbral e y_pred < umbral
        for r in fn_list:
            assert r["y_real"] >= umbral
            assert r["y_pred"] < umbral

        # Todos los FP deben tener y_real < umbral e y_pred >= umbral
        for r in fp_list:
            assert r["y_real"] < umbral
            assert r["y_pred"] >= umbral


class TestConfigurableUmbral:
    """Verifica que el umbral τ sea configurable vía variable de entorno o parámetro."""

    def test_umbral_configurable_via_env_var(self, monkeypatch):
        """Verifica que el umbral se configure dinámicamente mediante UMBRAL_CONFLICTO."""
        monkeypatch.setenv("UMBRAL_CONFLICTO", "13.0")
        _, meta = build_error_dataframe(seed=42)
        assert meta["umbral"] == 13.0

    def test_border_cases_adapt_to_custom_umbral(self, error_data):
        """Verifica que get_border_cases recalcule los casos frontera al pasar un umbral explícito."""
        test_df, _ = error_data
        custom_umbral = 12.0
        border_12 = get_border_cases(test_df, umbral=custom_umbral)

        for r in border_12["falsos_negativos"]:
            assert r["y_real"] >= custom_umbral
            assert r["y_pred"] < custom_umbral

        for r in border_12["falsos_positivos"]:
            assert r["y_real"] < custom_umbral
            assert r["y_pred"] >= custom_umbral


class TestRunErrorAnalysisPersistence:
    """Verifica la ejecución y exportación JSON."""

    def test_run_error_analysis_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "diagnostico.json"

            result = run_error_analysis(
                data_path=DEFAULT_DATA,
                model_path=DEFAULT_MODEL,
                materias_path=DEFAULT_MATERIAS,
                out_json=out_json,
                seed=42,
            )

            assert out_json.exists()
            assert out_json.stat().st_size > 0

            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)

            assert "estadisticas_generales" in saved
            assert "top_10_errores" in saved
            assert "casos_frontera" in saved
            assert len(saved["top_10_errores"]) == 10
