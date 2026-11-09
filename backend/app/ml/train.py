"""
train.py — Entrenamiento y comparación de modelos de regresión.

Compara 3 modelos para predecir cantidad de estudiantes afectados:
1. Regresión Lineal
2. Árbol de Decisión
3. K-Nearest Neighbors (KNN)

Uso:
    python -m app.ml.train --data data/pares_materias.csv --out models/modelo.joblib
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

# Rutas por defecto relativas al backend
BASE_DIR = Path(__file__).resolve().parent  # backend/app/ml
BACKEND_DIR = BASE_DIR.parents[1]           # backend
DEFAULT_DATA = BACKEND_DIR / "data" / "pares_materias.csv"
DEFAULT_MODEL = BACKEND_DIR / "models" / "modelo.joblib"

# Asegurar que backend/ esté en el sys.path si se invoca desde fuera de backend/
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from app.ml.features import build_preprocessor, load_dataset, split_data

# Semilla por defecto para reproducibilidad
DEFAULT_SEED = 42


def get_candidate_models(seed: int = DEFAULT_SEED) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Define los 3 modelos candidatos con sus pipelines y grillas de hiperparámetros."""
    candidates = {
        "Regresión Lineal": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("regressor", LinearRegression()),
            ]),
            {
                # Modelo baseline sin hiperparámetros libres
            },
        ),
        "Árbol de Decisión": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("regressor", DecisionTreeRegressor(random_state=seed)),
            ]),
            {
                "regressor__max_depth": [3, 5, 8, 12, None],
                "regressor__min_samples_split": [2, 5, 10],
                "regressor__min_samples_leaf": [1, 2, 4],
            },
        ),
        "KNN": (
            Pipeline([
                ("preprocessor", build_preprocessor()),
                ("regressor", KNeighborsRegressor()),
            ]),
            {
                "regressor__n_neighbors": [3, 5, 7, 9, 11, 15],
                "regressor__weights": ["uniform", "distance"],
                "regressor__p": [1, 2],
            },
        ),
    }
    return candidates


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = DEFAULT_SEED,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Ajusta los hiperparámetros con 5-Fold CV sobre Train y evalúa en Validación.

    Garantía metodológica:
    - El ajuste de hiperparámetros (CV) ocurre **exclusivamente sobre Train**.
    - La comparación final se realiza sobre el conjunto de **Validación** (117 pares).
    - El conjunto de **Test** queda estrictamente preservado para T10.
    """
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    candidates = get_candidate_models(seed=seed)

    results: Dict[str, Dict[str, Any]] = {}

    for name, (pipeline, param_grid) in candidates.items():
        if param_grid:
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_rmse = -grid_search.best_score_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {"modelo": "mínimos cuadrados ordinarios (OLS)"}
            # Estimar CV score para la regresión lineal
            cv_scores = []
            for train_idx, val_idx in cv.split(X_train):
                fold_pipe = Pipeline([
                    ("preprocessor", build_preprocessor()),
                    ("regressor", LinearRegression()),
                ])
                fold_pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                y_pred_fold = fold_pipe.predict(X_train.iloc[val_idx])
                cv_scores.append(root_mean_squared_error(y_train.iloc[val_idx], y_pred_fold))
            cv_rmse = float(np.mean(cv_scores))

        # Evaluación sobre Train (para análisis de sesgo/varianza)
        y_pred_train = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        train_rmse = root_mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)

        # Evaluación sobre Validación (criterio decisorio)
        y_pred_val = best_model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_rmse = root_mean_squared_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)

        results[name] = {
            "model": best_model,
            "best_params": best_params,
            "cv_rmse": cv_rmse,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
        }

    # Seleccionar el mejor modelo según menor RMSE en validación (y mayor R2)
    best_name = min(results, key=lambda k: results[k]["val_rmse"])

    return {
        "results": results,
        "best_model_name": best_name,
        "best_model": results[best_name]["model"],
    }


def format_table(results_dict: Dict[str, Dict[str, Any]]) -> str:
    """Genera la tabla comparativa de métricas en formato texto/markdown."""
    lines = [
        "| Modelo | CV RMSE (Train) | Val MAE | Val RMSE | Val R² | Train R² |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, r in results_dict.items():
        lines.append(
            f"| **{name}** | {r['cv_rmse']:.3f} | {r['val_mae']:.3f} | "
            f"{r['val_rmse']:.3f} | {r['val_r2']:.4f} | {r['train_r2']:.4f} |"
        )
    return "\n".join(lines)


def train(
    data_path: str = str(DEFAULT_DATA),
    out_path: str = str(DEFAULT_MODEL),
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Punto de entrada principal para entrenar, comparar y persistir el mejor modelo."""
    print("=" * 70)
    print("          ENTRENAMIENTO Y COMPARACIÓN DE MODELOS (T09)")
    print("=" * 70)

    # Resolución robusta de la ruta de datos si se pasa relativa
    if not os.path.exists(data_path):
        alt_path = BACKEND_DIR / data_path
        if alt_path.exists():
            data_path = str(alt_path)

    print(f"Cargando dataset desde: {data_path}")

    df = load_dataset(data_path)
    split = split_data(df, seed=seed)

    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]

    print(f"Conjunto de Train: {len(X_train)} pares | Validación: {len(X_val)} pares")
    print(f"Ajustando hiperparámetros con 5-Fold CV sobre Train...\n")

    comparison = train_and_compare(X_train, y_train, X_val, y_val, seed=seed)
    results = comparison["results"]
    best_name = comparison["best_model_name"]
    best_model = comparison["best_model"]

    print("--- Tabla Comparativa sobre Validación ---")
    print(format_table(results))
    print()

    print("--- Mejores hiperparámetros encontrados ---")
    for name, r in results.items():
        print(f"• {name}: {r['best_params']}")
    print()

    print("--- Modelo Seleccionado ---")
    best_res = results[best_name]
    print(f"🏆 Ganador: **{best_name}**")
    print(f"   Val MAE:  {best_res['val_mae']:.3f} estudiantes")
    print(f"   Val RMSE: {best_res['val_rmse']:.3f} estudiantes")
    print(f"   Val R²:   {best_res['val_r2']:.4f}")
    print()

    # Asegurar directorio de salida
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    joblib.dump(best_model, out_path)
    print(f"Modelo serializado exitosamente en: {out_path}")
    print("=" * 70)

    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento y comparación de modelos de regresión."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Ruta al dataset CSV de pares de materias.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_MODEL),
        help="Ruta donde guardar el modelo seleccionado.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla para reproducibilidad.",
    )
    args = parser.parse_args()
    train(data_path=args.data, out_path=args.out, seed=args.seed)
