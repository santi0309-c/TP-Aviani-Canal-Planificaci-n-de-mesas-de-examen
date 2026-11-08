"""
test_graph_build.py — Pruebas unitarias para la construcción del grafo de conflictos (Ticket T13).
"""

import networkx as nx
import pytest

from app.graph.build import (
    build_conflict_graph,
    graph_to_dict,
    load_mock_predictions,
    load_materias_catalog,
)


def test_isolated_nodes_included():
    """Verifica que las materias sin aristas existan como nodos aislados en el grafo."""
    materias = [f"M{i:02d}" for i in range(1, 11)]  # 10 materias
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 10.0},
        {"materia_1": "M03", "materia_2": "M04", "conflicto_predicho": 5.0},
    ]

    # Con umbral alto (50), ninguna predicción supera el umbral
    G = build_conflict_graph(predictions, umbral=50.0, materias=materias)

    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 0
    # Todas las materias están aisladas (grado 0)
    for m in materias:
        assert G.degree[m] == 0


def test_edge_weight_attribute():
    """Verifica que las aristas tengan el atributo 'weight' con el valor de conflicto correcto."""
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 24.5},
        {"materia_1": "M02", "materia_3": "M03", "conflicto_predicho": 12.0},
    ]

    G = build_conflict_graph(predictions, umbral=15.0)

    assert G.has_edge("M01", "M02")
    assert not G.has_edge("M02", "M03")
    assert G["M01"]["M02"]["weight"] == 24.5
    assert G["M01"]["M02"]["conflicto"] == 24.5


def test_monotonicity_increasing_threshold_never_adds_edges():
    """
    CRITERIO DE ACEPTACIÓN T13:
    Subir el umbral nunca aumenta la cantidad de aristas (monotonía no creciente).
    """
    mock_preds = load_mock_predictions()
    materias = [f"M{i:02d}" for i in range(1, 41)]

    umbrales = [0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 70.0, 100.0, 200.0]
    cantidades_aristas = []
    conjuntos_aristas = []

    for tau in umbrales:
        G = build_conflict_graph(mock_preds, umbral=tau, materias=materias)
        cantidades_aristas.append(G.number_of_edges())
        conjuntos_aristas.append({frozenset(edge) for edge in G.edges})
        # La cantidad de nodos siempre se mantiene en 40
        assert G.number_of_nodes() == 40

    for i in range(len(umbrales) - 1):
        assert conjuntos_aristas[i + 1] <= conjuntos_aristas[i], (
            f"Al subir el umbral de {umbrales[i]} a {umbrales[i + 1]} "
            "se agregaron aristas o se conservó una arista que debía filtrarse."
        )
        assert cantidades_aristas[i + 1] <= cantidades_aristas[i]


def test_graph_is_undirected():
    """Verifica que el grafo sea no dirigido y simétrico."""
    predictions = [
        {"materia_1": "M01", "materia_2": "M05", "conflicto_predicho": 30.0}
    ]
    G = build_conflict_graph(predictions, umbral=10.0)

    assert not G.is_directed()
    assert G.has_edge("M01", "M05")
    assert G.has_edge("M05", "M01")


def test_flexible_dictionary_keys():
    """Verifica soporte para distintos formatos de claves (materia_id_1, cant_estudiantes_afectados)."""
    predictions = [
        {
            "materia_id_1": "M10",
            "materia_id_2": "M20",
            "cant_estudiantes_afectados": 18,
        }
    ]
    G = build_conflict_graph(predictions, umbral=15.0)

    assert G.has_edge("M10", "M20")
    assert G["M10"]["M20"]["weight"] == 18.0


def test_graph_serialization_to_dict():
    """Verifica que graph_to_dict devuelva la estructura requerida por la API."""
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 20.0}
    ]
    materias_catalog = [{"id": "M01", "nombre": "Álgebra I"}, {"id": "M02", "nombre": "Física I"}]
    G = build_conflict_graph(predictions, umbral=15.0, materias=materias_catalog)

    data = graph_to_dict(G, umbral=15.0)

    assert data["umbral"] == 15.0
    assert data["cantidad_nodos"] == 2
    assert data["cantidad_aristas"] == 1
    assert len(data["nodos"]) == 2
    assert len(data["aristas"]) == 1
    assert data["aristas"][0]["peso"] == 20.0
