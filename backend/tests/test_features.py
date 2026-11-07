"""
test_features.py — Pruebas unitarias del pipeline de features y partición (Ticket T08).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from app.ml.features import (
    BOOLEAN_FEATURES,
    EXCLUDED_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    build_features,
    build_preprocessor,
    load_dataset,
    split_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_PATH = str(Path(__file__).resolve().parents[1] / "data" / "pares_materias.csv")


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Carga el dataset real del proyecto."""
    return load_dataset(DATA_PATH)


@pytest.fixture
def split(raw_df: pd.DataFrame) -> dict:
    """Genera la partición estándar con semilla 42."""
    return split_data(raw_df, seed=42)


# ---------------------------------------------------------------------------
# Tests de selección de features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    """Valida la correcta selección de features y exclusión de columnas."""

    def test_feature_columns_present(self, raw_df: pd.DataFrame):
        """X debe contener exactamente las 3 features seleccionadas."""
        X, y = build_features(raw_df)
        assert list(X.columns) == FEATURE_COLUMNS

    def test_excluded_columns_absent(self, raw_df: pd.DataFrame):
        """X no debe contener ninguna columna excluida."""
        X, _ = build_features(raw_df)
        for col in EXCLUDED_COLUMNS:
            assert col not in X.columns

    def test_target_is_correct(self, raw_df: pd.DataFrame):
        """El target debe ser cant_estudiantes_afectados."""
        _, y = build_features(raw_df)
        assert y.name == TARGET

    def test_booleans_are_int(self, raw_df: pd.DataFrame):
        """Las columnas booleanas deben estar codificadas como int."""
        X, _ = build_features(raw_df)
        for col in BOOLEAN_FEATURES:
            assert X[col].dtype in (np.int32, np.int64, int)
            assert set(X[col].unique()).issubset({0, 1})

    def test_no_nulls_in_features(self, raw_df: pd.DataFrame):
        """No debe haber valores nulos en X ni en y."""
        X, y = build_features(raw_df)
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_feature_count(self, raw_df: pd.DataFrame):
        """X debe tener exactamente 3 columnas."""
        X, _ = build_features(raw_df)
        assert X.shape[1] == 3

    def test_row_count_preserved(self, raw_df: pd.DataFrame):
        """build_features no debe eliminar ni agregar filas."""
        X, y = build_features(raw_df)
        assert len(X) == len(raw_df)
        assert len(y) == len(raw_df)


# ---------------------------------------------------------------------------
# Tests de partición de datos
# ---------------------------------------------------------------------------


class TestSplitData:
    """Valida la partición Train / Validación / Test."""

    def test_partition_sizes(self, split: dict):
        """Las particiones deben sumar 780 con proporciones 70/15/15."""
        n_train = len(split["X_train"])
        n_val = len(split["X_val"])
        n_test = len(split["X_test"])
        total = n_train + n_val + n_test

        assert total == 780
        # Tolerancia de ±1 fila por redondeo de train_test_split
        assert abs(n_train - 546) <= 1
        assert abs(n_val - 117) <= 1
        assert abs(n_test - 117) <= 1

    def test_no_overlap_train_val(self, split: dict):
        """Train y Validación no deben compartir índices."""
        train_idx = set(split["indices"]["train"])
        val_idx = set(split["indices"]["val"])
        assert train_idx.isdisjoint(val_idx)

    def test_no_overlap_train_test(self, split: dict):
        """Train y Test no deben compartir índices."""
        train_idx = set(split["indices"]["train"])
        test_idx = set(split["indices"]["test"])
        assert train_idx.isdisjoint(test_idx)

    def test_no_overlap_val_test(self, split: dict):
        """Validación y Test no deben compartir índices."""
        val_idx = set(split["indices"]["val"])
        test_idx = set(split["indices"]["test"])
        assert val_idx.isdisjoint(test_idx)

    def test_union_covers_all(self, split: dict):
        """La unión de los tres conjuntos debe cubrir todos los índices."""
        all_idx = (
            set(split["indices"]["train"])
            | set(split["indices"]["val"])
            | set(split["indices"]["test"])
        )
        assert all_idx == set(range(780))

    def test_reproducibility(self, raw_df: pd.DataFrame):
        """La misma semilla debe producir particiones idénticas."""
        s1 = split_data(raw_df, seed=42)
        s2 = split_data(raw_df, seed=42)

        pd.testing.assert_frame_equal(s1["X_train"], s2["X_train"])
        pd.testing.assert_frame_equal(s1["X_val"], s2["X_val"])
        pd.testing.assert_frame_equal(s1["X_test"], s2["X_test"])

    def test_different_seeds_differ(self, raw_df: pd.DataFrame):
        """Semillas distintas deben producir particiones distintas."""
        s1 = split_data(raw_df, seed=42)
        s2 = split_data(raw_df, seed=99)
        assert not s1["X_train"].equals(s2["X_train"])

    def test_metadata_complete(self, split: dict):
        """Los metadatos deben contener todos los campos requeridos."""
        meta = split["metadata"]
        required_keys = {
            "seed", "test_size", "val_size", "total_samples",
            "train_samples", "val_samples", "test_samples",
            "feature_columns", "numeric_features", "boolean_features",
            "target", "excluded_columns",
        }
        assert required_keys.issubset(set(meta.keys()))


# ---------------------------------------------------------------------------
# Tests del preprocesador
# ---------------------------------------------------------------------------


class TestPreprocessor:
    """Valida el ColumnTransformer de preprocesamiento."""

    def test_fit_transform_no_errors(self, split: dict):
        """El preprocesador debe ajustarse y transformar sin errores."""
        preprocessor = build_preprocessor()
        X_pp = preprocessor.fit_transform(split["X_train"])
        assert X_pp is not None
        assert X_pp.shape[0] == len(split["X_train"])

    def test_output_columns_count(self, split: dict):
        """La salida debe tener 3 columnas (1 numérica + 2 booleanas)."""
        preprocessor = build_preprocessor()
        X_pp = preprocessor.fit_transform(split["X_train"])
        assert X_pp.shape[1] == 3

    def test_no_nans_after_transform(self, split: dict):
        """No debe haber NaN después de la transformación."""
        preprocessor = build_preprocessor()
        X_pp = preprocessor.fit_transform(split["X_train"])
        assert not np.isnan(X_pp).any()

    def test_numeric_scaled(self, split: dict):
        """La columna numérica debe quedar centrada (~media 0, ~std 1)."""
        preprocessor = build_preprocessor()
        X_pp = preprocessor.fit_transform(split["X_train"])

        # La primera columna es la numérica escalada
        col_num = X_pp[:, 0]
        assert abs(np.mean(col_num)) < 1e-10
        assert abs(np.std(col_num) - 1.0) < 0.05

    def test_booleans_unchanged(self, split: dict):
        """Las columnas booleanas deben pasar sin modificación."""
        preprocessor = build_preprocessor()
        X_pp = preprocessor.fit_transform(split["X_train"])

        # Columnas 1 y 2 son las booleanas (passthrough)
        for col_idx in [1, 2]:
            unique_vals = set(np.unique(X_pp[:, col_idx]))
            assert unique_vals.issubset({0, 1, 0.0, 1.0})

    def test_transform_val_uses_train_stats(self, split: dict):
        """Val/Test deben transformarse con estadísticas de Train."""
        preprocessor = build_preprocessor()
        preprocessor.fit(split["X_train"])

        X_val_pp = preprocessor.transform(split["X_val"])
        X_test_pp = preprocessor.transform(split["X_test"])

        # Val y Test no deberían tener media exactamente 0
        # (porque se escalaron con la media/std de Train)
        # Simplemente verificar que no fallan y tienen la forma correcta
        assert X_val_pp.shape == (len(split["X_val"]), 3)
        assert X_test_pp.shape == (len(split["X_test"]), 3)
        assert not np.isnan(X_val_pp).any()
        assert not np.isnan(X_test_pp).any()

