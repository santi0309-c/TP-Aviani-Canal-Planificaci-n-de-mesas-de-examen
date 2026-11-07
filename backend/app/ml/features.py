"""
features.py — Preparación de datos, selección de features y partición (Ticket T08).

Módulo responsable de:
- Seleccionar las features predictoras y el target de regresión.
- Construir el ColumnTransformer de preprocesamiento (escalado + codificación).
- Partir el dataset en Train / Validación / Test sin fuga de información.

Uso interno:
    from app.ml.features import load_dataset, build_features, split_data, build_preprocessor
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------------------
# Constantes de configuración del pipeline
# ---------------------------------------------------------------------------

TARGET = "cant_estudiantes_afectados"

FEATURE_COLUMNS: List[str] = [
    "cant_estudiantes_compartidos_historico",
    "mismo_anio",
    "misma_carrera",
]

NUMERIC_FEATURES: List[str] = ["cant_estudiantes_compartidos_historico"]
BOOLEAN_FEATURES: List[str] = ["mismo_anio", "misma_carrera"]

# Columnas que se excluyen explícitamente del entrenamiento
EXCLUDED_COLUMNS: List[str] = [
    "materia_id_1",     # Identificador institucional sin poder predictivo
    "materia_id_2",     # Identificador institucional sin poder predictivo
    "hubo_conflicto",   # Binarización del target → Target Leakage crítico
]

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# ---------------------------------------------------------------------------
# Carga del dataset
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Carga el CSV del dataset y valida que contenga las columnas requeridas.

    Parameters
    ----------
    path : str
        Ruta al archivo ``pares_materias.csv``.

    Returns
    -------
    pd.DataFrame
        DataFrame con las 7 columnas esperadas y 780 filas (para N=40).

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    ValueError
        Si faltan columnas requeridas en el CSV.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el dataset en: {path}")

    df = pd.read_csv(path)

    required = set(FEATURE_COLUMNS + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {missing}"
        )

    return df


# ---------------------------------------------------------------------------
# Selección de features y target
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Selecciona las features predictoras y el target del DataFrame crudo.

    Se excluyen explícitamente:
    - ``materia_id_1``, ``materia_id_2``: identificadores sin poder de
      generalización (usarlos causaría memorización / sobreajuste).
    - ``hubo_conflicto``: es una binarización directa del target
      (Target Leakage crítico).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo con las 7 columnas del dataset.

    Returns
    -------
    X : pd.DataFrame
        Matriz de features con columnas:
        ``cant_estudiantes_compartidos_historico``, ``mismo_anio``,
        ``misma_carrera``.
    y : pd.Series
        Vector target ``cant_estudiantes_afectados``.
    """
    X = df[FEATURE_COLUMNS].copy()

    # Garantizar que las booleanas sean int (0/1) para compatibilidad
    # con todos los estimadores de scikit-learn
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)

    y = df[TARGET].copy()

    return X, y


# ---------------------------------------------------------------------------
# Preprocesador (ColumnTransformer)
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer de preprocesamiento.

    Transformaciones aplicadas:
    - **Numéricas** (``cant_estudiantes_compartidos_historico``):
      ``StandardScaler`` para centrar en media 0 y desvío 1, mitigando
      el impacto de la asimetría positiva y los valores extremos.
    - **Booleanas** (``mismo_anio``, ``misma_carrera``):
      ``passthrough`` — ya están codificadas como int {0, 1} por
      ``build_features()``, no requieren transformación adicional.

    Returns
    -------
    ColumnTransformer
        Transformador listo para ``fit`` sobre Train y ``transform``
        sobre Val/Test.

    Notes
    -----
    Este transformador se ajusta (``fit``) **únicamente** con los datos
    de entrenamiento para evitar fuga de información (data leakage).
    Los conjuntos de validación y test solo reciben ``transform``.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bool", "passthrough", BOOLEAN_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Partición de datos (Train / Validación / Test)
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
) -> Dict[str, Any]:
    """Parte el dataset en Train / Validación / Test de forma determinista.

    La partición se realiza en dos etapas con ``train_test_split``:
    1. Separar Test (15%) del resto (85%).
    2. Separar Validación (15% del total ≈ 17.65% del resto) de Train (70%).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame completo (780 filas para N=40).
    seed : int
        Semilla para reproducibilidad exacta.
    test_size : float
        Proporción del conjunto de test (default: 0.15).
    val_size : float
        Proporción del conjunto de validación (default: 0.15).

    Returns
    -------
    dict
        Diccionario con las claves:
        - ``X_train``, ``X_val``, ``X_test``: DataFrames de features.
        - ``y_train``, ``y_val``, ``y_test``: Series de target.
        - ``indices``: dict con los índices originales de cada partición.
        - ``metadata``: dict con semilla, proporciones y columnas usadas.
    """
    X, y = build_features(df)

    # Etapa 1: separar Test del resto
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
    )

    # Etapa 2: separar Validación de Train
    # val_size relativo al total → proporción sobre el subconjunto restante
    val_relative = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_relative,
        random_state=seed,
    )

    # Metadata para reproducibilidad
    metadata: Dict[str, Any] = {
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "total_samples": len(df),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "boolean_features": BOOLEAN_FEATURES,
        "target": TARGET,
        "excluded_columns": EXCLUDED_COLUMNS,
    }

    indices = {
        "train": X_train.index.tolist(),
        "val": X_val.index.tolist(),
        "test": X_test.index.tolist(),
    }

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "indices": indices,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Persistencia de metadatos
# ---------------------------------------------------------------------------

def save_split_metadata(metadata: Dict[str, Any], path: str) -> None:
    """Guarda los metadatos de la partición en un archivo JSON.

    Parameters
    ----------
    metadata : dict
        Diccionario de metadatos retornado por ``split_data()``.
    path : str
        Ruta de destino para el archivo JSON.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadatos de partición guardados en: {path}")


# ---------------------------------------------------------------------------
# Función de conveniencia: pipeline completo de preparación
# ---------------------------------------------------------------------------

def prepare_data(
    data_path: str,
    seed: int = DEFAULT_SEED,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    pd.Series, pd.Series, pd.Series,
    ColumnTransformer, Dict[str, Any],
]:
    """Pipeline completo: carga, selecciona features, parte y preprocesa.

    El preprocesador se ajusta (``fit``) **solo con Train**. Los conjuntos
    de Validación y Test son transformados con las estadísticas de Train.

    Parameters
    ----------
    data_path : str
        Ruta al archivo CSV del dataset.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    X_train_pp, X_val_pp, X_test_pp : np.ndarray
        Matrices de features preprocesadas.
    y_train, y_val, y_test : pd.Series
        Vectores target (sin transformar).
    preprocessor : ColumnTransformer
        Transformador ajustado con Train (para uso en predicción).
    metadata : dict
        Metadatos de la partición.
    """
    df = load_dataset(data_path)
    split = split_data(df, seed=seed)

    preprocessor = build_preprocessor()

    # fit SOLO con Train, transform sobre los tres conjuntos
    X_train_pp = preprocessor.fit_transform(split["X_train"])
    X_val_pp = preprocessor.transform(split["X_val"])
    X_test_pp = preprocessor.transform(split["X_test"])

    return (
        X_train_pp,
        X_val_pp,
        X_test_pp,
        split["y_train"],
        split["y_val"],
        split["y_test"],
        preprocessor,
        split["metadata"],
    )
