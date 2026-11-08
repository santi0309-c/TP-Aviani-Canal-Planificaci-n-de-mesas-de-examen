"""Pruebas del validador de coloreo (Ticket T15)."""

import networkx as nx
import pytest

from app.graph.coloring import validate_coloring


@pytest.mark.parametrize(
    "coloring",
    [
        {"M01": 0, "M02": 0},
        {"M01": 0},
        {"M01": -1, "M02": 1},
        {"M01": 0.5, "M02": 1},
    ],
    ids=["adyacentes-mismo-turno", "nodo-sin-turno", "turno-negativo", "turno-no-entero"],
)
def test_validator_rejects_invalid_colorings(coloring):
    """El validador debe rechazar coloreos inválidos construidos explícitamente."""
    graph = nx.Graph()
    graph.add_nodes_from(["M01", "M02"])
    graph.add_edge("M01", "M02")

    assert validate_coloring(graph, coloring) is False


def test_validator_accepts_valid_coloring():
    """Dos materias conectadas en turnos distintos forman un coloreo válido."""
    graph = nx.Graph()
    graph.add_edge("M01", "M02")

    assert validate_coloring(graph, {"M01": 0, "M02": 1}) is True


def test_validator_accepts_empty_graph():
    """Un grafo vacío es válido con el coloreo vacío."""
    assert validate_coloring(nx.Graph(), {}) is True
