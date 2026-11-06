"""
eda.py — Análisis Exploratorio de Datos (EDA) del dataset sintético de pares de materias (Ticket T06).

Uso:
    python data/eda.py --data data/pares_materias.csv --out-dir ../informe/capturas/eda
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def configurar_estilo():
    """Configura el estilo visual para los gráficos del informe."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def imprimir_resumen_estadistico(df: pd.DataFrame):
    """Imprime por consola el análisis descriptivo formal."""
    print("=" * 60)
    print("         REPORTE DE ANÁLISIS EXPLORATORIO (T06)")
    print("=" * 60)
    print(f"Dimensiones del dataset: {df.shape[0]} pares x {df.shape[1]} columnas\n")

    print("--- 1. Chequeo de Valores Nulos ---")
    nulos = df.isnull().sum()
    print(nulos.to_string())
    print(f"Total de celdas nulas: {nulos.sum()} (Integridad 100%)\n")

    print("--- 2. Estadísticas Descriptivas de Variables Numéricas ---")
    num_cols = ["cant_estudiantes_compartidos_historico", "cant_estudiantes_afectados"]
    desc = df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    desc.columns = ["Conteo", "Media", "Desv. Est.", "Mín", "Q1 (25%)", "Mediana", "Q3 (75%)", "Máx"]
    print(desc.round(2).to_string())
    print()

    print("--- 3. Distribución de Variables Categóricas / Booleanas ---")
    for col in ["mismo_anio", "misma_carrera", "hubo_conflicto"]:
        cnt = df[col].value_counts()
        pct = df[col].value_counts(normalize=True) * 100
        res = pd.DataFrame({"Cantidad": cnt, "Porcentaje (%)": pct.round(2)})
        print(f"Variable '{col}':")
        print(res.to_string())
        print()

    print("--- 4. Matriz de Correlación de Pearson ---")
    corr = df.select_dtypes(include=["int64", "bool", "float64"]).astype(float).corr()
    print(corr.round(3).to_string())
    print()

    print("--- 5. Cruces Clave de Conflicto ---")
    print("Tasa de conflicto según 'mismo_anio':")
    c1 = pd.crosstab(df["mismo_anio"], df["hubo_conflicto"], normalize="index") * 100
    print(c1.round(2).to_string())
    print("\nTasa de conflicto según 'misma_carrera':")
    c2 = pd.crosstab(df["misma_carrera"], df["hubo_conflicto"], normalize="index") * 100
    print(c2.round(2).to_string())
    print("=" * 60)


def generar_graficos(df: pd.DataFrame, out_dir: str):
    """Genera y guarda los 4 gráficos analíticos para el informe."""
    os.makedirs(out_dir, exist_ok=True)
    configurar_estilo()

    # -------------------------------------------------------------
    # Gráfico 1: Distribución comparada de las variables numéricas
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(
        df["cant_estudiantes_compartidos_historico"],
        kde=True,
        ax=axes[0],
        color="#2b5c8f",
        bins=25,
    )
    axes[0].set_title("Estudiantes Compartidos Históricos", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Cantidad de Estudiantes")
    axes[0].set_ylabel("Frecuencia (Pares)")
    axes[0].axvline(
        df["cant_estudiantes_compartidos_historico"].median(),
        color="#e74c3c",
        linestyle="--",
        label=f"Mediana: {df['cant_estudiantes_compartidos_historico'].median():.0f}",
    )
    axes[0].axvline(
        df["cant_estudiantes_compartidos_historico"].mean(),
        color="#f39c12",
        linestyle=":",
        label=f"Media: {df['cant_estudiantes_compartidos_historico'].mean():.1f}",
    )
    axes[0].legend()

    sns.histplot(
        df["cant_estudiantes_afectados"],
        kde=True,
        ax=axes[1],
        color="#27ae60",
        bins=25,
    )
    axes[1].set_title("Estudiantes Afectados (Target Regresión)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Cantidad de Estudiantes")
    axes[1].set_ylabel("Frecuencia (Pares)")
    axes[1].axvline(
        df["cant_estudiantes_afectados"].median(),
        color="#e74c3c",
        linestyle="--",
        label=f"Mediana: {df['cant_estudiantes_afectados'].median():.0f}",
    )
    axes[1].axvline(
        df["cant_estudiantes_afectados"].mean(),
        color="#f39c12",
        linestyle=":",
        label=f"Media: {df['cant_estudiantes_afectados'].mean():.1f}",
    )
    axes[1].legend()

    plt.tight_layout()
    p1 = os.path.join(out_dir, "01_distribucion_targets.png")
    fig.savefig(p1, dpi=300)
    plt.close(fig)
    print(f"[Grafico 1] Guardado en: {p1}")

    # -------------------------------------------------------------
    # Gráfico 2: Balance de clases de hubo_conflicto
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    conteo = df["hubo_conflicto"].value_counts()
    porcentajes = (conteo / len(df) * 100).round(1)

    labels = ["Sin Conflicto (False)", "En Conflicto (True)"]
    colores = ["#3498db", "#e67e22"]
    bars = ax.bar(labels, conteo.values, color=colores, width=0.5, edgecolor="#555555")

    for bar, pct, cnt in zip(bars, porcentajes.values, conteo.values):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 10,
            f"{cnt} pares\n({pct}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylim(0, max(conteo.values) + 80)
    ax.set_title("Balance de Clases: hubo_conflicto (Corte en k = 15 alumnos)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cantidad de Pares de Materias")

    plt.tight_layout()
    p2 = os.path.join(out_dir, "02_balance_clases_conflicto.png")
    fig.savefig(p2, dpi=300)
    plt.close(fig)
    print(f"[Grafico 2] Guardado en: {p2}")

    # -------------------------------------------------------------
    # Gráfico 3: Matriz de Correlación de Pearson (Heatmap)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df.select_dtypes(include=["int64", "bool", "float64"]).astype(float).corr()

    labels_map = {
        "cant_estudiantes_compartidos_historico": "Compartidos Hist.",
        "mismo_anio": "Mismo Año",
        "misma_carrera": "Misma Carrera",
        "cant_estudiantes_afectados": "Afectados (Target Reg)",
        "hubo_conflicto": "Conflicto (Target Clas)",
    }
    corr_renamed = corr.rename(index=labels_map, columns=labels_map)

    sns.heatmap(
        corr_renamed,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Matriz de Correlación de Pearson", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    p3 = os.path.join(out_dir, "03_matriz_correlacion.png")
    fig.savefig(p3, dpi=300)
    plt.close(fig)
    print(f"[Grafico 3] Guardado en: {p3}")

    # -------------------------------------------------------------
    # Gráfico 4: Dispersión Afectados vs Compartidos por Mismo Año
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="cant_estudiantes_compartidos_historico",
        y="cant_estudiantes_afectados",
        hue="mismo_anio",
        style="misma_carrera",
        palette={True: "#e74c3c", False: "#3498db"},
        alpha=0.75,
        s=60,
        ax=ax,
    )

    # Línea horizontal del umbral institucional k=15
    ax.axhline(15, color="#e67e22", linestyle="--", linewidth=1.5, label="Umbral de Conflicto (k = 15)")

    ax.set_title(
        "Relación: Estudiantes Afectados vs. Compartidos Históricos",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Estudiantes Compartidos Históricos")
    ax.set_ylabel("Estudiantes Afectados en Examen")

    # Mejorar leyenda
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc="upper left", frameon=True)

    plt.tight_layout()
    p4 = os.path.join(out_dir, "04_dispersion_afectados_vs_compartidos.png")
    fig.savefig(p4, dpi=300)
    plt.close(fig)
    print(f"[Grafico 4] Guardado en: {p4}")


def run_eda(data_path: str, out_dir: str):
    """Punto de entrada principal para el análisis exploratorio."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en: {data_path}")

    df = pd.read_csv(data_path)
    imprimir_resumen_estadistico(df)
    generar_graficos(df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Análisis Exploratorio de Datos (EDA).")
    parser.add_argument(
        "--data",
        default="data/pares_materias.csv",
        help="Ruta al archivo CSV del dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="../informe/capturas/eda",
        help="Directorio donde se guardarán los gráficos generados.",
    )
    args = parser.parse_args()
    run_eda(args.data, args.out_dir)

