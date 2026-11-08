"""
error_analysis.py — Análisis de errores del modelo sobre el conjunto de test.

Revisa en qué pares de materias se equivoca más el modelo:
- Pares con mayor error absoluto (|y - ŷ|).
- Falsos positivos y falsos negativos según el umbral τ.
- Diferencias por año de cursada y por carrera.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from app.ml.features import load_dataset, split_data

# Rutas estándar del proyecto
BASE_DIR = Path(__file__).resolve().parent      # backend/app/ml
BACKEND_DIR = BASE_DIR.parents[1]               # backend
REPO_ROOT = BACKEND_DIR.parent                  # raíz

DEFAULT_DATA = BACKEND_DIR / "data" / "pares_materias.csv"
DEFAULT_MODEL = BACKEND_DIR / "models" / "modelo.joblib"
DEFAULT_MATERIAS = BACKEND_DIR / "data" / "materias.json"
DEFAULT_OUT_JSON = BACKEND_DIR / "models" / "diagnostico_errores.json"


def get_default_umbral() -> float:
    """
    Obtiene el umbral institucional de conflicto.
    Prioriza la variable de entorno UMBRAL_CONFLICTO si está definida;
    en su defecto, utiliza 15.0 como umbral reglamentario estándar.
    """
    env_val = os.environ.get("UMBRAL_CONFLICTO")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            pass
    return 15.0


DEFAULT_UMBRAL = get_default_umbral()


def load_materias_dict(json_path: Path) -> Dict[str, str]:
    """Carga el catálogo de materias como diccionario {id: nombre}."""
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {m["id"]: m["nombre"] for m in data}


def build_error_dataframe(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    materias_path: Path = DEFAULT_MATERIAS,
    seed: int = 42,
    umbral: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Genera el DataFrame enriquecido con predicciones, residuos y diagnósticos sobre Test."""
    if umbral is None:
        umbral = get_default_umbral()
    else:
        umbral = float(umbral)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

    # Si no existe el modelo preentrenado (entorno CI), entrenarlo automáticamente
    if not model_path.exists():
        print(f"Modelo no encontrado en '{model_path}'. Entrenando modelo automáticamente...")
        from app.ml.train import train
        train(data_path=str(data_path), out_path=str(model_path), seed=seed)

    df_raw = load_dataset(str(data_path))
    split = split_data(df_raw, seed=seed)
    X_test, y_test = split["X_test"], split["y_test"]

    model = joblib.load(str(model_path))
    y_pred = model.predict(X_test)

    # Recuperar registros del DataFrame original correspondientes a Test
    test_df = df_raw.loc[X_test.index].copy()

    # Nombres de materias
    materias_map = load_materias_dict(materias_path)
    test_df["materia_1_nombre"] = test_df["materia_id_1"].map(materias_map).fillna(test_df["materia_id_1"])
    test_df["materia_2_nombre"] = test_df["materia_id_2"].map(materias_map).fillna(test_df["materia_id_2"])

    # Métricas individuales de residuo
    test_df["y_real"] = y_test.astype(int)
    test_df["y_pred"] = np.round(y_pred, 2)
    test_df["residuo"] = np.round(test_df["y_real"] - test_df["y_pred"], 2)
    test_df["error_abs"] = np.round(np.abs(test_df["residuo"]), 2)

    # Binarización y clasificación de errores según umbral τ
    test_df["conflicto_real"] = test_df["y_real"] >= umbral
    test_df["conflicto_pred"] = test_df["y_pred"] >= umbral

    test_df["es_falso_negativo"] = test_df["conflicto_real"] & (~test_df["conflicto_pred"])
    test_df["es_falso_positivo"] = (~test_df["conflicto_real"]) & test_df["conflicto_pred"]
    test_df["es_acierto"] = test_df["conflicto_real"] == test_df["conflicto_pred"]

    # Segmentación por volumen de estudiantes compartidos
    bins = [-1, 15, 50, 1000]
    bin_labels = ["Bajo (<15)", "Medio [15, 50)", "Alto (≥50)"]
    test_df["volumen_compartidos"] = pd.cut(
        test_df["cant_estudiantes_compartidos_historico"],
        bins=bins,
        labels=bin_labels,
    )

    metadata = {
        "n_test": len(test_df),
        "umbral": umbral,
        "semilla": seed,
    }

    return test_df, metadata


def compute_error_statistics(test_df: pd.DataFrame) -> Dict[str, Any]:
    """Calcula estadísticas generales de sesgo y desglose por factores."""
    residuos = test_df["residuo"].to_numpy()
    media_res = float(np.mean(residuos))
    desvio_res = float(np.std(residuos, ddof=1))

    # Conteo de sesgo direccional
    subestimados = int(np.sum(residuos > 0))  # y_real > y_pred
    sobreestimados = int(np.sum(residuos < 0))  # y_real < y_pred
    exactos = int(np.sum(residuos == 0))
    total = len(residuos)

    sesgo_dict = {
        "media_residuos": round(media_res, 4),
        "desvio_residuos": round(desvio_res, 4),
        "cant_subestimados": subestimados,
        "pct_subestimados": round((subestimados / total) * 100, 2),
        "cant_sobreestimados": sobreestimados,
        "pct_sobreestimados": round((sobreestimados / total) * 100, 2),
        "cant_exactos": exactos,
    }

    # Desglose por mismo_anio
    desglose_anio = {}
    for val, grp in test_df.groupby("mismo_anio", observed=False):
        desglose_anio[bool(val)] = {
            "count": int(len(grp)),
            "mean": round(float(grp["error_abs"].mean()), 4),
            "std": round(float(grp["error_abs"].std(ddof=1)), 4),
        }

    # Desglose por misma_carrera
    desglose_carrera = {}
    for val, grp in test_df.groupby("misma_carrera", observed=False):
        desglose_carrera[bool(val)] = {
            "count": int(len(grp)),
            "mean": round(float(grp["error_abs"].mean()), 4),
            "std": round(float(grp["error_abs"].std(ddof=1)), 4),
        }

    # Desglose por volumen
    desglose_vol = {}
    for val, grp in test_df.groupby("volumen_compartidos", observed=False):
        if len(grp) > 0:
            desglose_vol[str(val)] = {
                "count": int(len(grp)),
                "mean": round(float(grp["error_abs"].mean()), 4),
                "std": round(float(grp["error_abs"].std(ddof=1)), 4),
            }

    return {
        "sesgo": sesgo_dict,
        "desglose_mismo_anio": desglose_anio,
        "desglose_misma_carrera": desglose_carrera,
        "desglose_volumen": desglose_vol,
    }


def get_top_errors(test_df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """Extrae el Top N de pares con mayor error absoluto |y - ŷ|."""
    top_df = test_df.sort_values(by="error_abs", ascending=False).head(n)
    records = []
    for _, r in top_df.iterrows():
        records.append({
            "par": f"{r['materia_id_1']}-{r['materia_id_2']}",
            "materia_1": r["materia_1_nombre"],
            "materia_2": r["materia_2_nombre"],
            "compartidos": int(r["cant_estudiantes_compartidos_historico"]),
            "mismo_anio": bool(r["mismo_anio"]),
            "misma_carrera": bool(r["misma_carrera"]),
            "y_real": int(r["y_real"]),
            "y_pred": float(r["y_pred"]),
            "residuo": float(r["residuo"]),
            "error_abs": float(r["error_abs"]),
        })
    return records


def get_border_cases(test_df: pd.DataFrame, umbral: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Devuelve los casos de clasificación errónea (Falsos Negativos y Falsos Positivos) en el umbral indicado."""
    if umbral is not None:
        umbral_val = float(umbral)
        conf_real = test_df["y_real"] >= umbral_val
        conf_pred = test_df["y_pred"] >= umbral_val
        fn_df = test_df[conf_real & (~conf_pred)].sort_values(by="error_abs", ascending=False)
        fp_df = test_df[(~conf_real) & conf_pred].sort_values(by="error_abs", ascending=False)
    else:
        fn_df = test_df[test_df["es_falso_negativo"]].sort_values(by="error_abs", ascending=False)
        fp_df = test_df[test_df["es_falso_positivo"]].sort_values(by="error_abs", ascending=False)

    def to_record_list(df_subset):
        return [
            {
                "par": f"{r['materia_id_1']}-{r['materia_id_2']}",
                "materia_1": r["materia_1_nombre"],
                "materia_2": r["materia_2_nombre"],
                "compartidos": int(r["cant_estudiantes_compartidos_historico"]),
                "mismo_anio": bool(r["mismo_anio"]),
                "misma_carrera": bool(r["misma_carrera"]),
                "y_real": int(r["y_real"]),
                "y_pred": float(r["y_pred"]),
                "residuo": float(r["residuo"]),
                "error_abs": float(r["error_abs"]),
            }
            for _, r in df_subset.iterrows()
        ]

    return {
        "falsos_negativos": to_record_list(fn_df),
        "falsos_positivos": to_record_list(fp_df),
    }


def run_error_analysis(
    data_path: Path = DEFAULT_DATA,
    model_path: Path = DEFAULT_MODEL,
    materias_path: Path = DEFAULT_MATERIAS,
    out_json: Path = DEFAULT_OUT_JSON,
    seed: int = 42,
    umbral: Optional[float] = None,
) -> Dict[str, Any]:
    """Punto de entrada principal para el diagnóstico de errores."""
    if umbral is None:
        umbral = get_default_umbral()
    else:
        umbral = float(umbral)

    print("=" * 70)
    print("          DIAGNÓSTICO Y ANÁLISIS DE ERRORES (T11)")
    print("=" * 70)

    test_df, meta = build_error_dataframe(
        data_path=data_path,
        model_path=model_path,
        materias_path=materias_path,
        seed=seed,
        umbral=umbral,
    )

    stats = compute_error_statistics(test_df)
    top_errors = get_top_errors(test_df, n=10)
    border_cases = get_border_cases(test_df, umbral=umbral)

    sesgo = stats["sesgo"]
    print(f"Total de pares evaluados en Test: {meta['n_test']} (Umbral τ = {meta['umbral']})")
    print(f"• Media de residuos: {sesgo['media_residuos']:+.4f} alumnos")
    print(f"• Desvío de residuos: {sesgo['desvio_residuos']:.4f} alumnos")
    print(f"• Subestimaciones (y > ŷ): {sesgo['cant_subestimados']} ({sesgo['pct_subestimados']}%)")
    print(f"• Sobreestimaciones (y < ŷ): {sesgo['cant_sobreestimados']} ({sesgo['pct_sobreestimados']}%)")
    print()

    print("--- TOP 5 PARES CON MAYOR DISCREPANCIA ---")
    for i, r in enumerate(top_errors[:5], start=1):
        print(f"{i}. [{r['par']}] {r['materia_1']} vs. {r['materia_2']}")
        print(f"   Real: {r['y_real']} | Predicho: {r['y_pred']:.2f} | Residuo: {r['residuo']:+.2f} | Comp: {r['compartidos']}")
    print()

    print(f"--- CASOS FRONTERA DE CLASIFICACIÓN (τ = {umbral}) ---")
    fn_list = border_cases["falsos_negativos"]
    fp_list = border_cases["falsos_positivos"]
    print(f"• Falsos Negativos ({len(fn_list)} casos: real ≥ {umbral}, predicho < {umbral}):")
    for r in fn_list:
        print(f"  - [{r['par']}] {r['materia_1']} vs. {r['materia_2']} → Real: {r['y_real']} | Pred: {r['y_pred']:.2f}")

    print(f"• Falsos Positivos ({len(fp_list)} casos: real < {umbral}, predicho ≥ {umbral}):")
    for r in fp_list:
        print(f"  - [{r['par']}] {r['materia_1']} vs. {r['materia_2']} → Real: {r['y_real']} | Pred: {r['y_pred']:.2f}")
    print()

    # Guardar diagnóstico en JSON
    diagnostico_completo = {
        "metadatos": meta,
        "estadisticas_generales": stats,
        "top_10_errores": top_errors,
        "casos_frontera": border_cases,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(diagnostico_completo, f, ensure_ascii=False, indent=2)

    print(f"Diagnóstico exportado a: {out_json}")
    print("=" * 70)

    return diagnostico_completo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de diagnóstico de errores del modelo de ML.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Ruta al dataset CSV.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ruta al modelo joblib.")
    parser.add_argument("--materias", default=str(DEFAULT_MATERIAS), help="Ruta al catálogo materias.json.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_JSON), help="Ruta del JSON de diagnóstico.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de partición.")
    parser.add_argument(
        "--umbral",
        type=float,
        default=None,
        help="Umbral de conflicto en estudiantes (por defecto toma variable UMBRAL_CONFLICTO o 15).",
    )
    args = parser.parse_args()

    umbral_final = args.umbral if args.umbral is not None else get_default_umbral()

    run_error_analysis(
        data_path=Path(args.data),
        model_path=Path(args.model),
        materias_path=Path(args.materias),
        out_json=Path(args.out),
        seed=args.seed,
        umbral=umbral_final,
    )
