"""
train.py — Entrenamiento, ajuste de hiperparámetros y comparación de modelos (Ticket T09).

Compara 3 familias de modelos de regresión:
1. Regresión Lineal (baseline paramétrico e interpretable)
2. Árbol de Decisión (no lineal con particiones ortogonales)
3. K-Nearest Neighbors (no paramétrico basado en distancias)

Uso por CLI:
    python -m app.ml.train --data data/pares_materias.csv --out models/modelo.joblib
"""

import argparse
import os
from typing import Any, Dict, Tuple

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
    """Define los 3 modelos candidatos con sus pipelines y grillas de hiperparámetros.

    Cada modelo se encapsula en un Pipeline con el ColumnTransformer de T08.
    """
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


def train(data_path: str, out_path: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Punto de entrada principal para entrenar, comparar y persistir el mejor modelo."""
    print("=" * 70)
    print("          ENTRENAMIENTO Y COMPARACIÓN DE MODELOS (T09)")
    print("=" * 70)
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

    print("--- Hiperparámetros Óptimos ---")
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
        default="data/pares_materias.csv",
        help="Ruta al dataset CSV de pares de materias.",
    )
    parser.add_argument(
        "--out",
        default="models/modelo.joblib",
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
