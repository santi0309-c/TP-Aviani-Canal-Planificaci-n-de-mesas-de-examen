"""
evaluate.py — Evaluación final del modelo elegido sobre el conjunto de test.

Evalúa el modelo (Regresión Lineal) sobre los 117 pares de test:
1. Métricas de regresión: MAE, RMSE, R².
2. Métricas de clasificación derivada con corte en umbral τ = 15: Accuracy, Precisión, Recall, F1.
3. Guarda el JSON en backend/models/metricas.json para que lo use el backend.
4. Genera gráficos en informe/capturas/evaluacion/.

Uso:
    python -m app.ml.evaluate --data data/pares_materias.csv --model models/modelo.joblib
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")  # Para que no abra ventanas de gráficos
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from app.ml.features import load_dataset, split_data

# Rutas predeterminadas robustas relativas a la ubicación del script
BASE_DIR = Path(__file__).resolve().parent  # backend/app/ml
BACKEND_DIR = BASE_DIR.parents[1]           # backend
REPO_ROOT = BACKEND_DIR.parent              # raíz del repositorio

DEFAULT_DATA = BACKEND_DIR / "data" / "pares_materias.csv"
DEFAULT_MODEL = BACKEND_DIR / "models" / "modelo.joblib"
DEFAULT_METRICS_OUT = BACKEND_DIR / "models" / "metricas.json"
DEFAULT_PLOTS_DIR = REPO_ROOT / "informe" / "capturas" / "evaluacion"
DEFAULT_UMBRAL = 15


def calcular_metricas(
    y_true: pd.Series,
    y_pred: np.ndarray,
    umbral: int = DEFAULT_UMBRAL,
    semilla: int = 42,
    nombre_modelo: str = "Regresión Lineal (OLS)",
) -> Dict[str, Any]:
    """Calcula métricas primarias de regresión y derivadas de clasificación."""
    # Métricas continuas de regresión
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Binarización institucional para clasificación
    y_true_bin = (y_true >= umbral).astype(int)
    y_pred_bin = (y_pred >= umbral).astype(int)

    acc = float(accuracy_score(y_true_bin, y_pred_bin))
    prec = float(precision_score(y_true_bin, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true_bin, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true_bin, y_pred_bin, zero_division=0))

    cm = confusion_matrix(y_true_bin, y_pred_bin)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    return {
        "metadatos": {
            "modelo": nombre_modelo,
            "semilla": semilla,
            "n_test": len(y_true),
        },
        "regresion": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
        },
        "clasificacion": {
            "umbral": umbral,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        },
        "matriz_confusion": {
            "verdaderos_negativos": tn,
            "falsos_positivos": fp,
            "falsos_negativos": fn,
            "verdaderos_positivos": tp,
            "total": len(y_true),
        },
    }


def generar_graficos(
    y_true: pd.Series,
    y_pred: np.ndarray,
    out_dir: Path,
    umbral: int = DEFAULT_UMBRAL,
) -> None:
    """Genera y guarda los 3 gráficos analíticos sobre el conjunto de test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # -------------------------------------------------------------
    # Gráfico 1: Predicho vs. Real (Scatter + Diagonal de Identidad)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        y_true,
        y_pred,
        color="#2b5c8f",
        alpha=0.75,
        edgecolors="w",
        s=70,
        label="Pares de Test (N=117)",
    )

    max_val = max(float(y_true.max()), float(y_pred.max())) + 5
    min_val = min(float(y_true.min()), float(y_pred.min())) - 2
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Predicción Ideal (y = x)")

    # Umbral de conflicto k = 15
    ax.axvline(umbral, color="#f39c12", linestyle=":", linewidth=1.5, label=f"Umbral de Conflicto (τ = {umbral})")
    ax.axhline(umbral, color="#f39c12", linestyle=":", linewidth=1.5)

    ax.set_title("Evaluación en Test: Afectados Predichos vs. Reales", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Estudiantes Afectados Reales ($y$)", fontsize=11)
    ax.set_ylabel(r"Estudiantes Afectados Predichos ($\hat{y}$)", fontsize=11)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    p1 = out_dir / "01_predicho_vs_real.png"
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"[Grafico 1] Guardado en: {p1}")

    # -------------------------------------------------------------
    # Gráfico 2: Matriz de Confusión Derivada (Heatmap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    y_true_bin = (y_true >= umbral).astype(int)
    y_pred_bin = (y_pred >= umbral).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)

    labels = [["VN", "FP"], ["FN", "VP"]]
    annot_text = np.empty_like(cm, dtype=object)
    total = np.sum(cm)
    for i in range(2):
        for j in range(2):
            cnt = cm[i, j]
            pct = cnt / total * 100
            annot_text[i, j] = f"{labels[i][j]}\n{cnt}\n({pct:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_text,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        xticklabels=["Sin Conflicto (<15)", "En Conflicto (≥15)"],
        yticklabels=["Sin Conflicto (<15)", "En Conflicto (≥15)"],
    )

    ax.set_title("Matriz de Confusión Derivada (Test, τ = 15)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Clase Predicha", fontsize=11)
    ax.set_ylabel("Clase Real", fontsize=11)

    plt.tight_layout()
    p2 = out_dir / "02_matriz_confusion_test.png"
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"[Grafico 2] Guardado en: {p2}")

    # -------------------------------------------------------------
    # Gráfico 3: Distribución de Errores Residuales (y - y_pred)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    residuos = y_true - y_pred

    sns.histplot(residuos, kde=True, color="#27ae60", bins=20, ax=ax)
    ax.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Error Cero ($e = 0$)")
    ax.axvline(residuos.mean(), color="#f39c12", linestyle=":", linewidth=1.5, label=f"Media: {residuos.mean():.2f}")

    ax.set_title(r"Distribución de Residuos en Test ($e = y - \hat{y}$)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Error Residual (Estudiantes)", fontsize=11)
    ax.set_ylabel("Frecuencia (Pares)", fontsize=11)
    ax.legend(frameon=True)

    plt.tight_layout()
    p3 = out_dir / "03_distribucion_residuos.png"
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"[Grafico 3] Guardado en: {p3}")


def evaluate(
    data_path: str = str(DEFAULT_DATA),
    model_path: str = str(DEFAULT_MODEL),
    metrics_out: str = str(DEFAULT_METRICS_OUT),
    plots_dir: str = str(DEFAULT_PLOTS_DIR),
    seed: int = 42,
    umbral: int = DEFAULT_UMBRAL,
) -> Dict[str, Any]:
    """Punto de entrada de evaluación sobre el conjunto de test."""
    print("=" * 65)
    print("          EVALUACIÓN FINAL SOBRE CONJUNTO DE TEST (T10)")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")
    if not os.path.exists(model_path):
        print(f"Modelo no encontrado en '{model_path}'. Entrenando modelo automáticamente...")
        from app.ml.train import train
        train(data_path=data_path, out_path=model_path, seed=seed)

    df = load_dataset(data_path)
    split = split_data(df, seed=seed)

    X_test, y_test = split["X_test"], split["y_test"]
    print(f"Conjunto de Test cargado: {len(X_test)} pares (15% del dataset)")

    model = joblib.load(model_path)
    print(f"Modelo cargado desde: {model_path}")

    # Inferencia
    y_pred = model.predict(X_test)

    # Cálculo de métricas
    metricas = calcular_metricas(
        y_true=y_test,
        y_pred=y_pred,
        umbral=umbral,
        semilla=seed,
    )

    print("\n--- Métricas de Regresión (Continuas) ---")
    reg = metricas["regresion"]
    print(f"• MAE:  {reg['mae']:.4f} estudiantes")
    print(f"• RMSE: {reg['rmse']:.4f} estudiantes")
    print(f"• R²:   {reg['r2']:.4f}")

    print("\n--- Métricas de Clasificación Derivada (τ = 15) ---")
    clas = metricas["clasificacion"]
    cm = metricas["matriz_confusion"]
    print(f"• Accuracy:  {clas['accuracy'] * 100:.2f}%")
    print(f"• Precision: {clas['precision'] * 100:.2f}%")
    print(f"• Recall:    {clas['recall'] * 100:.2f}%")
    print(f"• F1-Score:  {clas['f1'] * 100:.2f}%")
    print(f"• Matriz:    VN={cm['verdaderos_negativos']}, FP={cm['falsos_positivos']}, "
          f"FN={cm['falsos_negativos']}, VP={cm['verdaderos_positivos']}")

    # Guardar métricas JSON
    out_dir_metrics = os.path.dirname(metrics_out)
    if out_dir_metrics:
        os.makedirs(out_dir_metrics, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas exitosamente en: {metrics_out}")

    # Guardar gráficos
    plots_path = Path(plots_dir)
    generar_graficos(y_true=y_test, y_pred=y_pred, out_dir=plots_path, umbral=umbral)
    print("=" * 65)

    return metricas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script de evaluación final sobre el conjunto de test."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Ruta al dataset CSV.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Ruta al modelo joblib.")
    parser.add_argument("--out", default=str(DEFAULT_METRICS_OUT), help="Ruta de salida JSON.")
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR), help="Directorio para gráficos.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de partición.")
    parser.add_argument("--umbral", type=int, default=DEFAULT_UMBRAL, help="Umbral de conflicto.")
    args = parser.parse_args()

    evaluate(
        data_path=args.data,
        model_path=args.model,
        metrics_out=args.out,
        plots_dir=args.plots_dir,
        seed=args.seed,
        umbral=args.umbral,
    )
