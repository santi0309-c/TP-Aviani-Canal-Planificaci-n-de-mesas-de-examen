"""
build.py — Construcción del grafo de conflictos a partir de las predicciones del modelo.

Cada materia es un nodo y agregamos una arista entre dos materias si la cantidad
estimada de estudiantes afectados supera el umbral configurado.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import pandas as pd

# Rutas por defecto basadas en la estructura del proyecto
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR.parents[1]
DEFAULT_DATA_CSV = BACKEND_DIR / "data" / "pares_materias.csv"
DEFAULT_MATERIAS_JSON = BACKEND_DIR / "data" / "materias.json"


def build_conflict_graph(
    predictions: Union[List[Dict[str, Any]], pd.DataFrame],
    umbral: float,
    materias: Optional[List[Union[str, Dict[str, Any]]]] = None,
) -> nx.Graph:
    """
    Construye un grafo no dirigido (nx.Graph) de conflictos entre materias.

    Parámetros:
        predictions: Lista de registros o DataFrame de pandas con las predicciones por par.
                     Acepta claves 'materia_1'/'materia_id_1', 'materia_2'/'materia_id_2'
                     y 'conflicto_predicho'/'cant_estudiantes_afectados'/'peso'/'weight'.
        umbral: Valor mínimo de conflicto estimado para crear una arista.
        materias: Lista opcional de materias (IDs o dicts con atributos) para asegurar
                  que todas las materias existan como nodos, incluso si quedan aisladas.

    Retorna:
        nx.Graph: Grafo no dirigido con las materias como nodos y aristas con atributo 'weight'.
    """
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.to_dict(orient="records")


    G = nx.Graph()
    umbral_float = float(umbral)

    # 1. Registrar todos los nodos (materias), garantizando que las aisladas no se omitan
    if materias is not None:
        for m in materias:
            if isinstance(m, dict):
                node_id = m.get("id") or m.get("materia_id")
                if node_id:
                    attrs = {k: v for k, v in m.items() if k not in ("id", "materia_id")}
                    G.add_node(str(node_id), **attrs)
            elif isinstance(m, str):
                G.add_node(m)

    # 2. Procesar predicciones e incorporar aristas si superan el umbral
    for pred in predictions:
        u = str(pred.get("materia_1") or pred.get("materia_id_1") or "")
        v = str(pred.get("materia_2") or pred.get("materia_id_2") or "")

        if not u or not v or u == v:
            continue

        # Asegurar que ambos nodos existan en el grafo
        if u not in G:
            G.add_node(u)
        if v not in G:
            G.add_node(v)

        # Extraer el valor de conflicto predicho
        conflicto = None
        for key in ("conflicto_predicho", "cant_estudiantes_afectados", "peso", "weight"):
            if key in pred:
                conflicto = float(pred[key])
                break

        if conflicto is None:
            continue

        # Crear arista si el conflicto es mayor o igual al umbral
        if conflicto >= umbral_float:
            G.add_edge(u, v, weight=conflicto, conflicto=conflicto)

    return G


def load_materias_catalog(json_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Carga el catálogo de materias desde materias.json si existe."""
    target_path = Path(json_path) if json_path else DEFAULT_MATERIAS_JSON
    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_mock_predictions(csv_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """
    Carga un conjunto de predicciones mock a partir del dataset sintético generado en T05.
    Útil para desacoplar el desarrollo del grafo mientras se entrena el modelo en T08–T12.
    """
    target_path = Path(csv_path) if csv_path else DEFAULT_DATA_CSV
    if not target_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de predicciones en: {target_path}")

    df = pd.read_csv(target_path)
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "materia_1": str(row["materia_id_1"]),
                "materia_2": str(row["materia_id_2"]),
                "conflicto_predicho": float(row["cant_estudiantes_afectados"]),
            }
        )
    return records


def graph_to_dict(G: nx.Graph, umbral: float) -> Dict[str, Any]:
    """
    Serializa el grafo a una estructura de diccionario compatible con schemas de la API.
    """
    nodos = [
        {"id": n, "nombre": G.nodes[n].get("nombre", n)}
        for n in G.nodes()
    ]
    aristas = [
        {
            "source": u,
            "target": v,
            "peso": float(data.get("weight", 0.0)),
        }
        for u, v, data in G.edges(data=True)
    ]
    return {
        "umbral": float(umbral),
        "cantidad_nodos": G.number_of_nodes(),
        "cantidad_aristas": G.number_of_edges(),
        "nodos": nodos,
        "aristas": aristas,
    }
