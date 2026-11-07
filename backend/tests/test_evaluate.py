"""
test_evaluate.py — Pruebas unitarias para la evaluación final en test (Ticket T10).
"""

import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ml.evaluate import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    calcular_metricas,
    evaluate,
    generar_graficos,
)


class TestCalcularMetricas:
    """Verifica el cálculo correcto de métricas de regresión y clasificación."""

    def test_calcular_metricas_structure(self):
        y_true = pd.Series([0, 10, 20, 30])
        y_pred = np.array([1.0, 9.5, 21.0, 29.0])

        metrics = calcular_metricas(y_true, y_pred, umbral=15)

        assert "metadatos" in metrics
        assert "regresion" in metrics
        assert "clasificacion" in metrics
        assert "matriz_confusion" in metrics

        reg = metrics["regresion"]
        assert "mae" in reg and "rmse" in reg and "r2" in reg
        assert reg["mae"] >= 0.0
        assert reg["rmse"] >= 0.0
        assert reg["r2"] <= 1.0

        clas = metrics["clasificacion"]
        assert "accuracy" in clas and "f1" in clas and "recall" in clas and "precision" in clas
        assert 0.0 <= clas["accuracy"] <= 1.0
        assert 0.0 <= clas["f1"] <= 1.0

    def test_confusion_matrix_sum(self):
        y_true = pd.Series([5, 12, 16, 25, 40])
        y_pred = np.array([4, 18, 14, 30, 38])

        metrics = calcular_metricas(y_true, y_pred, umbral=15)
        cm = metrics["matriz_confusion"]

        total = cm["verdaderos_negativos"] + cm["falsos_positivos"] + cm["falsos_negativos"] + cm["verdaderos_positivos"]
        assert total == len(y_true)


class TestEvaluateExecution:
    """Verifica la ejecución de punta a punta del script de evaluación."""

    def test_evaluate_runs_and_creates_json_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = os.path.join(tmpdir, "metricas.json")
            plots_dir = os.path.join(tmpdir, "plots")

            # Garantizar que el modelo exista (en CI *.joblib está ignorado en git)
            if DEFAULT_MODEL.exists():
                model_to_use = str(DEFAULT_MODEL)
            else:
                from app.ml.train import train
                model_to_use = os.path.join(tmpdir, "modelo.joblib")
                train(data_path=str(DEFAULT_DATA), out_path=model_to_use, seed=42)

            metrics = evaluate(
                data_path=str(DEFAULT_DATA),
                model_path=model_to_use,
                metrics_out=out_json,
                plots_dir=plots_dir,
                seed=42,
            )

            # Verificar retorno
            assert metrics["metadatos"]["n_test"] == 117
            assert metrics["regresion"]["r2"] > 0.95
            assert metrics["regresion"]["mae"] < 2.5
            assert metrics["clasificacion"]["f1"] > 0.85

            # Verificar archivo JSON
            assert os.path.exists(out_json)
            with open(out_json, "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved == metrics

            # Verificar generación de los 3 gráficos
            plots_path = Path(plots_dir)
            p1 = plots_path / "01_predicho_vs_real.png"
            p2 = plots_path / "02_matriz_confusion_test.png"
            p3 = plots_path / "03_distribucion_residuos.png"

            assert p1.exists() and p1.stat().st_size > 0
            assert p2.exists() and p2.stat().st_size > 0
            assert p3.exists() and p3.stat().st_size > 0
