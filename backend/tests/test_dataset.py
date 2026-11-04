"""
test_dataset.py — Pruebas unitarias para el generador del dataset sintético (Ticket T05).
"""

import os
import tempfile
import pandas as pd
import pytest

from data.generate_dataset import generate


def test_dataset_reproducibility():
    """Verifica que la misma semilla genere exactamente el mismo dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "run1.csv")
        path2 = os.path.join(tmpdir, "run2.csv")

        df1 = generate(seed=42, n_materias=40, out_path=path1, save_materias_info=False)
        df2 = generate(seed=42, n_materias=40, out_path=path2, save_materias_info=False)

        pd.testing.assert_frame_equal(df1, df2)


def test_dataset_different_seeds():
    """Verifica que distintas semillas generen datos diferentes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "run1.csv")
        path2 = os.path.join(tmpdir, "run2.csv")

        df1 = generate(seed=42, n_materias=20, out_path=path1, save_materias_info=False)
        df2 = generate(seed=99, n_materias=20, out_path=path2, save_materias_info=False)

        assert not df1.equals(df2)


def test_dataset_shape_and_columns():
    """Verifica la cantidad esperada de pares y las columnas requeridas."""
    n_materias = 40
    expected_pairs = n_materias * (n_materias - 1) // 2  # C(40, 2) = 780

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "dataset.csv")
        df = generate(seed=42, n_materias=n_materias, out_path=out_path, save_materias_info=False)

        assert len(df) == expected_pairs
        expected_columns = [
            "materia_id_1",
            "materia_id_2",
            "cant_estudiantes_compartidos_historico",
            "mismo_anio",
            "misma_carrera",
            "cant_estudiantes_afectados",
            "hubo_conflicto",
        ]
        assert list(df.columns) == expected_columns
        assert df.isnull().sum().sum() == 0


def test_dataset_consistency_and_bounds():
    """Verifica cotas físicas: no negatividad y afectados <= compartidos."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "dataset.csv")
        df = generate(seed=42, n_materias=40, out_path=out_path, save_materias_info=False)

        # No hay valores negativos
        assert (df["cant_estudiantes_compartidos_historico"] >= 0).all()
        assert (df["cant_estudiantes_afectados"] >= 0).all()

        # Los afectados nunca pueden superar a los compartidos históricos
        assert (df["cant_estudiantes_afectados"] <= df["cant_estudiantes_compartidos_historico"]).all()

        # Consistencia del target de clasificación (umbral k = 15)
        conflicto_calculado = df["cant_estudiantes_afectados"] >= 15
        assert (df["hubo_conflicto"] == conflicto_calculado).all()


def test_no_self_pairs_or_duplicates():
    """Verifica que no haya pares de una materia consigo misma ni pares duplicados."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "dataset.csv")
        df = generate(seed=42, n_materias=30, out_path=out_path, save_materias_info=False)

        # Sin auto-pares
        assert (df["materia_id_1"] != df["materia_id_2"]).all()

        # Sin duplicados
        pares = list(zip(df["materia_id_1"], df["materia_id_2"]))
        assert len(pares) == len(set(pares))

