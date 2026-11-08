"""
predict.py — Carga del modelo entrenado y predicción de conflictos (Ticket T12).

Módulo responsable de:
- Cargar el pipeline entrenado (ColumnTransformer + Regresión Lineal) con patrón Singleton / Caché.
- Predecir la cantidad continua de estudiantes afectados para pares de materias.
- Exponer predecir_pares() para consumo directo del grafo (T13) y endpoints de la API (T17, T18).
- Servir como herramienta CLI para generar y exportar predicciones.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Rutas predeterminadas robustas relativas a la ubicación del script
BASE_DIR = Path(__file__).resolve().parent  # backend/app/ml
BACKEND_DIR = BASE_DIR.parents[1]           # backend
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "pares_materias.csv"
DEFAULT_MODEL_PATH = BACKEND_DIR / "models" / "modelo.joblib"

# Asegurar que backend/ esté en el sys.path si se invoca desde fuera de backend/
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import joblib
import numpy as np
import pandas as pd

from app.ml.features import FEATURE_COLUMNS, load_dataset

# Variable global para almacenamiento en caché del modelo (Patrón Singleton en memoria)
_CACHED_MODEL: Optional[Any] = None
_CACHED_MODEL_PATH: Optional[str] = None


def clear_model_cache() -> None:
    """Limpia el modelo almacenado en caché forzando una nueva lectura de disco en la próxima llamada."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    _CACHED_MODEL = None
    _CACHED_MODEL_PATH = None


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    reload: bool = False,
    auto_train: bool = True,
) -> Any:
    """
    Carga el pipeline entrenado desde disco utilizando persistencia con joblib y caché en memoria.

    Garantiza que el archivo .joblib se lea una sola vez en el ciclo de vida del proceso
    (ideal para FastAPI y llamadas repetidas desde endpoints), a menos que se indique reload=True.

    Parámetros:
        model_path: Ruta al archivo .joblib. Si es None, utiliza DEFAULT_MODEL_PATH.
        reload: Si es True, fuerza la recarga desde disco ignorando la caché.
        auto_train: Si es True y el modelo no existe, intenta entrenarlo automáticamente
                    a partir del dataset por defecto. Si es False, lanza FileNotFoundError.

    Retorna:
        Pipeline de scikit-learn entrenado y listo para inferencia.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    # Si la ruta es relativa y no existe en el directorio de trabajo actual,
    # intentar resolverla respecto a backend/
    if not target_path.exists() and not target_path.is_absolute():
        alt_path = BACKEND_DIR / target_path
        if alt_path.exists():
            target_path = alt_path

    resolved_str = str(target_path.resolve())

    # Retornar desde caché si ya está cargado y coincide la ruta
    if not reload and _CACHED_MODEL is not None and _CACHED_MODEL_PATH == resolved_str:
        return _CACHED_MODEL

    # Si el archivo no existe, verificar si se permite auto-entrenamiento o lanzar error
    if not target_path.exists():
        if not auto_train:
            raise FileNotFoundError(f"No se encontró el modelo en: {target_path}")

        if DEFAULT_DATA_PATH.exists():
            print(f"⚠️ Modelo no encontrado en '{target_path}'. Entrenando automáticamente...")
            from app.ml.train import train as train_model

            target_path.parent.mkdir(parents=True, exist_ok=True)
            train_model(data_path=str(DEFAULT_DATA_PATH), out_path=str(target_path), seed=42)
        else:
            raise FileNotFoundError(
                f"No se encontró el modelo en '{target_path}' ni el dataset en '{DEFAULT_DATA_PATH}'."
            )

    model = joblib.load(target_path)
    _CACHED_MODEL = model
    _CACHED_MODEL_PATH = resolved_str
    return model


def predict(model: Any, X: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]) -> np.ndarray:
    """
    Genera predicciones continuas de estudiantes afectados a partir del modelo y las features.

    Valida explícitamente la presencia de las features requeridas para evitar errores
    crípticos de scikit-learn si la entrada no tiene las columnas correctas.

    Parámetros:
        model: Pipeline de scikit-learn o estimador compatible con .predict().
        X: Matriz, lista de diccionarios o DataFrame con las features predictoras.

    Retorna:
        np.ndarray unidimensional con las predicciones acotadas en [0, inf).
    """
    if isinstance(X, list):
        X = pd.DataFrame(X)

    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"La matriz NumPy debe tener exactamente {len(FEATURE_COLUMNS)} columnas correspondientes a "
                f"{FEATURE_COLUMNS}, pero se recibieron {X.shape[1]} columnas."
            )
        X_input = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    elif isinstance(X, pd.DataFrame):
        missing_cols = [col for col in FEATURE_COLUMNS if col not in X.columns]
        if missing_cols:
            raise ValueError(
                f"El DataFrame no contiene las columnas requeridas para la predicción. "
                f"Columnas faltantes: {missing_cols}. Columnas esperadas: {FEATURE_COLUMNS}."
            )
        X_input = X[FEATURE_COLUMNS]
    else:
        raise TypeError(
            f"Tipo de entrada no soportado para inferencia: {type(X)}. "
            f"Se espera pd.DataFrame, np.ndarray o list[dict]."
        )

    preds = model.predict(X_input)
    preds = np.asarray(preds, dtype=float)

    # Acotar en 0 (no pueden existir estudiantes afectados negativos)
    return np.clip(preds, 0.0, None)


def predecir_pares(
    df: Optional[pd.DataFrame] = None,
    data_path: Optional[Union[str, Path]] = None,
    model: Optional[Any] = None,
    as_dict: bool = True,
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Predice la cantidad continua de estudiantes afectados para cada par de materias.

    Diseñada para ser consumida directamente por build_conflict_graph() en T13
    y por los endpoints de la API (GET /conflictos y POST /cronograma).

    Parámetros:
        df: DataFrame opcional con los pares. Si es None, se lee desde data_path o dataset default.
        data_path: Ruta al archivo CSV con los pares.
        model: Instancia del modelo entrenado. Si es None, se invoca load_model().
        as_dict: Si es True, retorna una lista de diccionarios estructurada:
                 [{'materia_1': 'M01', 'materia_2': 'M02', 'conflicto_predicho': 35.42, ...}]
                 Si es False, retorna el DataFrame enriquecido con la columna 'conflicto_predicho'.

    Retorna:
        Lista de diccionarios o DataFrame con las estimaciones de conflicto por par.
    """
    # 1. Resolver el conjunto de datos de entrada
    if df is None:
        target_data = Path(data_path) if data_path else DEFAULT_DATA_PATH
        if not target_data.exists() and not target_data.is_absolute():
            alt_path = BACKEND_DIR / target_data
            if alt_path.exists():
                target_data = alt_path

        if not target_data.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {target_data}")

        data_df = load_dataset(str(target_data))
    else:
        data_df = df.copy()

    # 2. Cargar modelo si no se proveyó
    if model is None:
        model = load_model()

    # 3. Inferencia de valores continuos
    raw_preds = predict(model, data_df)

    # Acotar superiormente con cant_estudiantes_compartidos_historico si la columna existe
    if "cant_estudiantes_compartidos_historico" in data_df.columns:
        compartidos = data_df["cant_estudiantes_compartidos_historico"].to_numpy(dtype=float)
        y_pred = np.minimum(raw_preds, compartidos)
    else:
        y_pred = raw_preds

    y_pred_rounded = np.round(y_pred, 2)

    # 4. Formatear salida según solicitud
    if not as_dict:
        result_df = data_df.copy()
        result_df["conflicto_predicho"] = y_pred_rounded
        result_df["cant_estudiantes_afectados_predicho"] = y_pred_rounded
        return result_df

    records: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(data_df.iterrows()):
        m1 = str(row.get("materia_1") or row.get("materia_id_1") or "")
        m2 = str(row.get("materia_2") or row.get("materia_id_2") or "")
        pred_val = float(y_pred_rounded[i])

        record: Dict[str, Any] = {
            "materia_1": m1,
            "materia_2": m2,
            "conflicto_predicho": pred_val,
            "cant_estudiantes_afectados": pred_val,
        }

        # Claves alternativas para máxima compatibilidad aguas abajo
        if "materia_id_1" in row:
            record["materia_id_1"] = str(row["materia_id_1"])
        if "materia_id_2" in row:
            record["materia_id_2"] = str(row["materia_id_2"])

        # Metadatos del par si están presentes
        if "cant_estudiantes_compartidos_historico" in row:
            record["cant_estudiantes_compartidos_historico"] = int(row["cant_estudiantes_compartidos_historico"])
        if "mismo_anio" in row:
            record["mismo_anio"] = bool(row["mismo_anio"])
        if "misma_carrera" in row:
            record["misma_carrera"] = bool(row["misma_carrera"])

        records.append(record)

    return records


def main() -> None:
    """Función de entrada para la ejecución por línea de comandos (CLI)."""
    parser = argparse.ArgumentParser(
        description="Genera predicciones de conflicto para pares de materias (Ticket T12)."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Ruta al dataset CSV de pares de materias.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta al modelo entrenado (.joblib).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta opcional para guardar las predicciones (JSON o CSV).",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="Cantidad de pares a mostrar en la vista previa (por defecto 10).",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("         MÓDULO DE PREDICCIÓN DE CONFLICTOS DE EXAMEN (T12)")
    print("=" * 70)
    print(f"Cargando modelo desde: {args.model}")
    model = load_model(args.model)
    print(f"Pipeline cargado exitosamente: {type(model).__name__}")
    if hasattr(model, "named_steps"):
        print(f"Pasos del pipeline: {list(model.named_steps.keys())}")

    print(f"\nGenerando predicciones sobre dataset: {args.data}")
    preds = predecir_pares(data_path=args.data, model=model, as_dict=True)
    print(f"✓ Total de pares evaluados: {len(preds)}")

    # Vista previa en tabla legible
    print(f"\n--- Vista previa de las primeras {min(args.head, len(preds))} predicciones ---")
    header = f"{'Par':<15} | {'Compartidos':<12} | {'Mismo Año':<10} | {'Misma Carrera':<14} | {'Conflicto Predicho':<18}"
    print(header)
    print("-" * len(header))
    for p in preds[:args.head]:
        par_str = f"{p['materia_1']} - {p['materia_2']}"
        comp = p.get("cant_estudiantes_compartidos_historico", "-")
        m_anio = "Sí" if p.get("mismo_anio") else "No"
        m_carr = "Sí" if p.get("misma_carrera") else "No"
        conf = f"{p['conflicto_predicho']:.2f}"
        print(f"{par_str:<15} | {str(comp):<12} | {m_anio:<10} | {m_carr:<14} | {conf:<18}")

    # Guardado opcional a archivo
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            df_out = pd.DataFrame(preds)
            df_out.to_csv(out_path, index=False)
            print(f"\n✓ Predicciones exportadas a CSV en: {out_path}")
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Predicciones exportadas a JSON en: {out_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
