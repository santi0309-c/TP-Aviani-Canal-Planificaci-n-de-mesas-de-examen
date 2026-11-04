"""
generate_dataset.py — Generador del dataset sintético de pares de materias.

Implementa las especificaciones formales definidas en informe/reglas_dataset.md (Ticket T04/T05).

Uso:
    python data/generate_dataset.py --seed 42 --n-materias 40 --out data/pares_materias.csv
"""

import argparse
import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Catálogo base de materias realistas para carreras de ingeniería
CATALOGO_MATERIAS = [
    # Año 1 (Tronco común / Básicas)
    {"nombre": "Álgebra y Geometría Analítica", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Análisis Matemático I", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Física I", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Química General", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Informática General", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Ingeniería y Sociedad", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Sistemas de Representación", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Fundamentos de Programación", "anio": 1, "carreras": ["C1", "C2", "C3"]},
    # Año 2 (Tronco común / Básicas avanzadas)
    {"nombre": "Análisis Matemático II", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Física II", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Probabilidad y Estadística", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Matemática Discreta", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Economía General", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Ciencia de los Materiales", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Estructuras de Datos", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    {"nombre": "Termodinámica Básica", "anio": 2, "carreras": ["C1", "C2", "C3"]},
    # Año 3 (Especialidades)
    {"nombre": "Arquitectura de Computadores", "anio": 3, "carreras": ["C1"]},
    {"nombre": "Sistemas Operativos", "anio": 3, "carreras": ["C1"]},
    {"nombre": "Investigación Operativa", "anio": 3, "carreras": ["C1", "C2"]},
    {"nombre": "Gestión de Procesos", "anio": 3, "carreras": ["C2"]},
    {"nombre": "Logística y Abastecimiento", "anio": 3, "carreras": ["C2"]},
    {"nombre": "Mecánica de Fluidos", "anio": 3, "carreras": ["C3"]},
    {"nombre": "Química Orgánica", "anio": 3, "carreras": ["C3"]},
    {"nombre": "Fenómenos de Transporte", "anio": 3, "carreras": ["C3"]},
    # Año 4 (Especialidades)
    {"nombre": "Redes y Telecomunicaciones", "anio": 4, "carreras": ["C1"]},
    {"nombre": "Bases de Datos", "anio": 4, "carreras": ["C1"]},
    {"nombre": "Ingeniería de Software", "anio": 4, "carreras": ["C1"]},
    {"nombre": "Control de Calidad", "anio": 4, "carreras": ["C2"]},
    {"nombre": "Seguridad e Higiene Industrial", "anio": 4, "carreras": ["C2"]},
    {"nombre": "Diseño de Plantas Industriales", "anio": 4, "carreras": ["C2", "C3"]},
    {"nombre": "Operaciones Unitarias", "anio": 4, "carreras": ["C3"]},
    {"nombre": "Reactores Químicos", "anio": 4, "carreras": ["C3"]},
    # Año 5 (Especialidades y Proyecto)
    {"nombre": "Inteligencia Artificial", "anio": 5, "carreras": ["C1"]},
    {"nombre": "Sistemas Distribuidos y Cloud", "anio": 5, "carreras": ["C1"]},
    {"nombre": "Gestión de Proyectos de Software", "anio": 5, "carreras": ["C1"]},
    {"nombre": "Dirección Estratégica", "anio": 5, "carreras": ["C2"]},
    {"nombre": "Simulación de Sistemas Productivos", "anio": 5, "carreras": ["C2"]},
    {"nombre": "Control de Procesos Químicos", "anio": 5, "carreras": ["C3"]},
    {"nombre": "Biotecnología Industrial", "anio": 5, "carreras": ["C3"]},
    {"nombre": "Proyecto Final Integrador", "anio": 5, "carreras": ["C1", "C2", "C3"]},
]

# Matrícula base esperada por año lectivo (estructura piramidal)
BASE_MATRICULA_POR_ANIO = {
    1: 250,
    2: 160,
    3: 90,
    4: 50,
    5: 30,
}


def generar_materias(n_materias: int, rng: np.random.Generator) -> List[Dict]:
    """Genera la lista de materias con sus atributos institucionales."""
    materias = []
    for i in range(n_materias):
        m_id = f"M{i + 1:02d}"
        if i < len(CATALOGO_MATERIAS):
            info = CATALOGO_MATERIAS[i]
            nombre = info["nombre"]
            anio = info["anio"]
            carreras = set(info["carreras"])
        else:
            anio = min(5, (i % 5) + 1)
            carrera_idx = (i % 3) + 1
            nombre = f"Materia Electiva {m_id}"
            carreras = {f"C{carrera_idx}"}

        base_alumnos = BASE_MATRICULA_POR_ANIO.get(anio, 50)
        ruido_tam = rng.normal(0, 15.0)
        tamanio = max(20, int(round(base_alumnos + ruido_tam)))

        materias.append(
            {
                "id": m_id,
                "nombre": nombre,
                "anio": anio,
                "carreras": carreras,
                "tamanio": tamanio,
            }
        )
    return materias


def calcular_par(
    m1: Dict, m2: Dict, rng: np.random.Generator, umbral_conflicto: int = 15
) -> Dict:
    """Aplica las reglas formales de T04 para un par de materias."""
    mismo_anio = bool(m1["anio"] == m2["anio"])
    misma_carrera = bool(m1["carreras"] & m2["carreras"])

    # Techo físico
    techo = min(m1["tamanio"], m2["tamanio"])

    # Factores de afinidad
    f_carrera = (
        float(rng.uniform(0.60, 0.85))
        if misma_carrera
        else float(rng.uniform(0.01, 0.05))
    )
    delta_anio = abs(m1["anio"] - m2["anio"])
    f_anio = 1.0 / (1.0 + 0.8 * delta_anio)

    # Estudiantes compartidos históricos
    mu_compartidos = techo * f_carrera * f_anio
    ruido_comp = float(rng.normal(0, 3.0))
    compartidos = int(np.clip(round(mu_compartidos + ruido_comp), 0, techo))

    # Target regresión: cant_estudiantes_afectados
    ruido_afect = float(rng.normal(0, 2.5))
    afectados_estim = (
        0.55 * compartidos
        + 6.0 * (1 if mismo_anio else 0)
        + 4.0 * (1 if misma_carrera else 0)
        + ruido_afect
    )
    afectados = int(np.clip(round(afectados_estim), 0, compartidos))

    # Target clasificación: hubo_conflicto
    hubo_conflicto = bool(afectados >= umbral_conflicto)

    return {
        "materia_id_1": m1["id"],
        "materia_id_2": m2["id"],
        "cant_estudiantes_compartidos_historico": compartidos,
        "mismo_anio": mismo_anio,
        "misma_carrera": misma_carrera,
        "cant_estudiantes_afectados": afectados,
        "hubo_conflicto": hubo_conflicto,
    }


def generate(
    seed: int = 42,
    n_materias: int = 40,
    out_path: str = "data/pares_materias.csv",
    save_materias_info: bool = True,
) -> pd.DataFrame:
    """Genera el dataset sintético reproducible y lo guarda en CSV."""
    rng = np.random.default_rng(seed)

    # 1. Generar materias
    materias = generar_materias(n_materias, rng)

    # 2. Generar todos los pares únicos sin repetición
    pares = []
    for m1, m2 in itertools.combinations(materias, 2):
        par_data = calcular_par(m1, m2, rng)
        pares.append(par_data)

    df = pd.DataFrame(pares)

    # Ordenar y verificar tipos de columnas
    columnas = [
        "materia_id_1",
        "materia_id_2",
        "cant_estudiantes_compartidos_historico",
        "mismo_anio",
        "misma_carrera",
        "cant_estudiantes_afectados",
        "hubo_conflicto",
    ]
    df = df[columnas]
    df["cant_estudiantes_compartidos_historico"] = df[
        "cant_estudiantes_compartidos_historico"
    ].astype(int)
    df["mismo_anio"] = df["mismo_anio"].astype(bool)
    df["misma_carrera"] = df["misma_carrera"].astype(bool)
    df["cant_estudiantes_afectados"] = df["cant_estudiantes_afectados"].astype(int)
    df["hubo_conflicto"] = df["hubo_conflicto"].astype(bool)

    # Asegurar que el directorio de salida exista
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Dataset generado exitosamente en: {out_path} ({len(df)} pares)")

    # Opcional: guardar catálogo de materias para uso de la API (/materias)
    if save_materias_info and out_dir:
        materias_path = os.path.join(out_dir, "materias.json")
        materias_serializables = [
            {
                "id": m["id"],
                "nombre": m["nombre"],
                "anio": m["anio"],
                "carreras": sorted(list(m["carreras"])),
                "tamanio": m["tamanio"],
            }
            for m in materias
        ]
        with open(materias_path, "w", encoding="utf-8") as f:
            json.dump(materias_serializables, f, ensure_ascii=False, indent=2)
        print(f"Catálogo de materias guardado en: {materias_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador del dataset sintético de pares de materias."
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    parser.add_argument(
        "--n-materias", type=int, default=40, help="Cantidad de materias a generar"
    )
    parser.add_argument(
        "--out", default="data/pares_materias.csv", help="Ruta del archivo CSV de salida"
    )
    args = parser.parse_args()
    generate(seed=args.seed, n_materias=args.n_materias, out_path=args.out)
