"""
error_analysis.py — Análisis y diagnóstico de errores del modelo sobre test (Ticket T11).

Diagnostica dónde y por qué falla el modelo de regresión:
1. Identifica los pares de materias con mayor error absoluto (|y - ŷ|).
2. Analiza los casos frontera de clasificación (Falsos Negativos y Falsos Positivos con umbral τ = 15).
3. Detecta patrones sistemáticos (sesgo general, desagregación por año, carrera y volumen).
4. Exporta un resumen diagnóstico consumible y genera reporte para el informe final.

Uso:
    python -m app.ml.error_analysis
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
DEFAULT_UMBRAL = 15


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
    umbral: int = DEFAULT_UMBRAL,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Genera el DataFrame enriquecido con predicciones, residuos y diagnósticos sobre Test."""
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
    """Calcula estadísticas agregadas y patrones de error sobre el conjunto de test."""
    residuos = test_df["residuo"]
    errores_abs = test_df["error_abs"]

    # 1. Sesgo general (Subestimación vs Sobreestimación)
    media_residuos = float(np.mean(residuos))
    desvio_residuos = float(np.std(residuos))
    subestimados = int((residuos > 0).sum())
    sobreestimados = int((residuos < 0).sum())
    exactos = int((residuos == 0).sum())
    n_total = len(test_df)

    sesgo_resumen = {
        "media_residuos": round(media_residuos, 4),
        "desvio_residuos": round(desvio_residuos, 4),
        "cant_subestimados": subestimados,
        "pct_subestimados": round((subestimados / n_total) * 100, 2),
        "cant_sobreestimados": sobreestimados,
        "pct_sobreestimados": round((sobreestimados / n_total) * 100, 2),
        "cant_exactos": exactos,
    }

    # 2. Desglose por mismo año
    mismo_anio_stat = (
        test_df.groupby("mismo_anio")["error_abs"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_dict(orient="index")
    )

    # 3. Desglose por misma carrera
    misma_carrera_stat = (
        test_df.groupby("misma_carrera")["error_abs"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_dict(orient="index")
    )

    # 4. Desglose por volumen de compartidos
    volumen_stat = (
        test_df.groupby("volumen_compartidos", observed=False)["error_abs"]
        .agg(["count", "mean", "std"])
        .round(4)
        .to_dict(orient="index")
    )

    return {
        "sesgo": sesgo_resumen,
        "desglose_mismo_anio": mismo_anio_stat,
        "desglose_misma_carrera": misma_carrera_stat,
        "desglose_volumen": volumen_stat,
    }


def get_top_errors(test_df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """Devuelve los N pares de materias con mayor error absoluto."""
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


def get_border_cases(test_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Devuelve los casos de clasificación errónea (Falsos Negativos y Falsos Positivos)."""
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
    umbral: int = DEFAULT_UMBRAL,
) -> Dict[str, Any]:
    """Punto de entrada principal para el diagnóstico de errores."""
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
    border_cases = get_border_cases(test_df)

    sesgo = stats["sesgo"]
    print(f"Total de pares evaluados en Test: {meta['n_test']}")
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

    print("--- CASOS FRONTERA DE CLASIFICACIÓN (τ = 15) ---")
    fn_list = border_cases["falsos_negativos"]
    fp_list = border_cases["falsos_positivos"]
    print(f"• Falsos Negativos ({len(fn_list)} casos: real ≥ 15, predicho < 15):")
    for r in fn_list:
        print(f"  - [{r['par']}] {r['materia_1']} vs. {r['materia_2']} → Real: {r['y_real']} | Pred: {r['y_pred']:.2f}")

    print(f"• Falsos Positivos ({len(fp_list)} casos: real < 15, predicho ≥ 15):")
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
    parser.add_argument("--umbral", type=int, default=DEFAULT_UMBRAL, help="Umbral de conflicto.")
    args = parser.parse_args()

    run_error_analysis(
        data_path=Path(args.data),
        model_path=Path(args.model),
        materias_path=Path(args.materias),
        out_json=Path(args.out),
        seed=args.seed,
        umbral=args.umbral,
    )

