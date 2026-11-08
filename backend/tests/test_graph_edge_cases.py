"""Pruebas de casos borde para construcción del grafo y coloreo (Ticket T16)."""

from itertools import combinations

from app.graph.build import build_conflict_graph
from app.graph.coloring import get_num_turnos, greedy_coloring, validate_coloring


def _predictions_for_edges(nodes, weight=10.0):
    return [
        {
            "materia_1": materia_1,
            "materia_2": materia_2,
            "conflicto_predicho": weight,
        }
        for materia_1, materia_2 in combinations(nodes, 2)
    ]


def test_fully_empty_graph_is_handled():
    """Un grafo sin materias ni predicciones debe colorearse sin errores."""
    graph = build_conflict_graph([], umbral=15.0, materias=[])

    coloring = greedy_coloring(graph)

    assert graph.number_of_nodes() == 0
    assert graph.number_of_edges() == 0
    assert coloring == {}
    assert get_num_turnos(coloring) == 0
    assert validate_coloring(graph, coloring)


def test_edgeless_graph_assigns_every_materia_to_one_turn():
    """Un grafo sin aristas debe conservar sus nodos y usar un único turno."""
    materias = ["M01", "M02", "M03", "M04"]
    graph = build_conflict_graph([], umbral=15.0, materias=materias)

    coloring = greedy_coloring(graph)

    assert graph.number_of_nodes() == len(materias)
    assert graph.number_of_edges() == 0
    assert set(coloring) == set(materias)
    assert set(coloring.values()) == {0}
    assert get_num_turnos(coloring) == 1
    assert validate_coloring(graph, coloring)


def test_catalog_as_dicts_preserves_materias_and_metadata():
    """El catálogo con objetos como los de materias.json debe crear nodos válidos."""
    materias = [
        {"id": "M01", "nombre": "Álgebra I", "anio": 1, "carreras": ["C1"]},
        {"id": "M02", "nombre": "Física I", "anio": 1, "carreras": ["C1"]},
        {"id": "M03", "nombre": "Química I", "anio": 1, "carreras": ["C2"]},
    ]
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 10.0}
    ]

    graph = build_conflict_graph(predictions, umbral=5.0, materias=materias)
    coloring = greedy_coloring(graph)

    assert set(graph.nodes) == {"M01", "M02", "M03"}
    assert graph.nodes["M01"]["nombre"] == "Álgebra I"
    assert graph.nodes["M01"]["anio"] == 1
    assert graph.nodes["M01"]["carreras"] == ["C1"]
    assert graph.has_edge("M01", "M02")
    assert graph.degree["M03"] == 0
    assert set(coloring) == set(graph.nodes)
    assert validate_coloring(graph, coloring)


def test_complete_graph_uses_one_turn_per_materia():
    """Un grafo completo K_n requiere exactamente n turnos distintos."""
    materias = ["M01", "M02", "M03", "M04", "M05"]
    predictions = _predictions_for_edges(materias)
    graph = build_conflict_graph(predictions, umbral=10.0, materias=materias)

    coloring = greedy_coloring(graph)

    assert graph.number_of_edges() == len(materias) * (len(materias) - 1) // 2
    assert get_num_turnos(coloring) == len(materias)
    assert set(coloring.values()) == set(range(len(materias)))
    assert validate_coloring(graph, coloring)


def test_isolated_materia_is_kept_and_colored():
    """Una materia sin conflictos debe permanecer en el grafo y recibir turno."""
    materias = ["M01", "M02", "M03"]
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 10.0}
    ]
    graph = build_conflict_graph(predictions, umbral=5.0, materias=materias)

    coloring = greedy_coloring(graph)

    assert set(graph.nodes) == set(materias)
    assert graph.degree["M03"] == 0
    assert set(coloring) == set(materias)
    assert validate_coloring(graph, coloring)


def test_threshold_above_all_predictions_removes_all_edges():
    """Un umbral mayor a todas las predicciones no debe crear aristas."""
    materias = ["M01", "M02", "M03"]
    predictions = [
        {"materia_1": "M01", "materia_2": "M02", "conflicto_predicho": 8.0},
        {"materia_1": "M02", "materia_2": "M03", "conflicto_predicho": 3.0},
    ]
    graph = build_conflict_graph(predictions, umbral=8.01, materias=materias)

    coloring = greedy_coloring(graph)

    assert graph.number_of_nodes() == len(materias)
    assert graph.number_of_edges() == 0
    assert set(coloring) == set(materias)
    assert get_num_turnos(coloring) == 1
    assert validate_coloring(graph, coloring)
