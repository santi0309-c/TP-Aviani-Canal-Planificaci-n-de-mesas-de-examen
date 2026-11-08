"""
test_coloring.py — Pruebas unitarias para el algoritmo de coloreo greedy (Ticket T14).
"""

import networkx as nx
import pytest

from app.graph.build import (
    build_conflict_graph,
    load_materias_catalog,
    load_mock_predictions,
)
from app.graph.coloring import (
    apply_coloring_to_graph,
    get_node_ordering,
    get_num_turnos,
    greedy_coloring,
    group_by_turno,
    validate_coloring,
)


def test_coloring_empty_graph():
    """Un grafo vacío debe retornar un coloreo vacío con 0 turnos."""
    G = nx.Graph()
    coloring = greedy_coloring(G)
    assert coloring == {}
    assert get_num_turnos(coloring) == 0
    assert validate_coloring(G, coloring)


def test_coloring_isolated_nodes():
    """Un grafo sin aristas debe asignar todas las materias al turno 0 (1 solo turno)."""
    G = nx.Graph()
    for i in range(1, 11):
        G.add_node(f"M{i:02d}")

    coloring = greedy_coloring(G)
    assert len(coloring) == 10
    assert all(color == 0 for color in coloring.values())
    assert get_num_turnos(coloring) == 1
    assert validate_coloring(G, coloring)


def test_coloring_complete_graph():
    """
    En un grafo completo K_n, cada par es adyacente.
    Debe requerir exactamente n colores distintos (número cromático chi = n).
    """
    n = 6
    G = nx.complete_graph(n)
    coloring = greedy_coloring(G)

    assert len(coloring) == n
    assert get_num_turnos(coloring) == n
    # Todos los colores deben ser únicos (0, 1, 2, ..., n-1)
    assert set(coloring.values()) == set(range(n))
    assert validate_coloring(G, coloring)


def test_coloring_bipartite_graph():
    """En un grafo bipartito conexo, el coloreo voraz debe usar a lo sumo 2 colores."""
    G = nx.complete_bipartite_graph(4, 4)
    coloring = greedy_coloring(G)

    assert get_num_turnos(coloring) <= 2
    assert validate_coloring(G, coloring)


def test_coloring_star_graph():
    """En un grafo estrella (1 centro + 8 hojas), debe usar exactamente 2 colores."""
    G = nx.star_graph(8)
    coloring = greedy_coloring(G)

    assert get_num_turnos(coloring) == 2
    assert validate_coloring(G, coloring)


def test_coloring_cycle_graphs():
    """Ciclo par C_4 (2 colores) y ciclo impar C_5 (3 colores)."""
    c4 = nx.cycle_graph(4)
    assert get_num_turnos(greedy_coloring(c4)) == 2
    assert validate_coloring(c4, greedy_coloring(c4))

    c5 = nx.cycle_graph(5)
    assert get_num_turnos(greedy_coloring(c5)) == 3
    assert validate_coloring(c5, greedy_coloring(c5))


def test_brooks_theorem_bound():
    """
    GARANTÍA TEÓRICA:
    La cantidad de turnos nunca debe superar Delta(G) + 1 (grado máximo + 1).
    """
    G = nx.erdos_renyi_graph(20, 0.3, seed=42)
    max_degree = max(dict(G.degree()).values())

    coloring = greedy_coloring(G)
    num_turnos = get_num_turnos(coloring)

    assert num_turnos <= max_degree + 1
    assert validate_coloring(G, coloring)


@pytest.mark.parametrize(
    ("node_count", "edge_probability", "seed"),
    [
        (0, 0.3, 0),
        (1, 0.5, 1),
        (12, 0.0, 2),
        (20, 0.15, 42),
        (30, 0.5, 73),
        (16, 1.0, 99),
    ],
)
def test_coloring_random_graphs_is_valid(node_count, edge_probability, seed):
    """El coloreo greedy debe ser válido en grafos aleatorios de distintas densidades."""
    graph = nx.gnp_random_graph(node_count, edge_probability, seed=seed)

    coloring = greedy_coloring(graph)

    assert set(coloring) == set(graph.nodes)
    assert validate_coloring(graph, coloring)
    assert all(isinstance(turno, int) and turno >= 0 for turno in coloring.values())
    assert all(coloring[u] != coloring[v] for u, v in graph.edges)


def test_coloring_real_project_dataset():
    """
    CRITERIO DE ACEPTACIÓN T14:
    Devuelve un coloreo válido para el grafo real del proyecto (40 materias, umbral tau = 15).
    """
    catalog = load_materias_catalog()
    preds = load_mock_predictions()
    G = build_conflict_graph(preds, umbral=15.0, materias=catalog)

    coloring = greedy_coloring(G)

    # 1. Todas las 40 materias deben tener turno asignado
    assert len(coloring) == 40
    for node in G.nodes():
        assert node in coloring

    # 2. El coloreo debe ser matemáticamente válido (sin aristas monocromáticas)
    assert validate_coloring(G, coloring)

    # 3. La cantidad de turnos es razonable (acotada por grado máximo + 1)
    max_degree = max(dict(G.degree()).values())
    num_turnos = get_num_turnos(coloring)
    assert 1 < num_turnos <= max_degree + 1


def test_coloring_determinism():
    """Dos llamadas consecutivas con el mismo grafo deben retornar exactamente el mismo coloreo."""
    preds = load_mock_predictions()
    G1 = build_conflict_graph(preds, umbral=20.0)
    G2 = build_conflict_graph(preds, umbral=20.0)

    c1 = greedy_coloring(G1)
    c2 = greedy_coloring(G2)

    assert c1 == c2


def test_node_ordering_welsh_powell():
    """Verifica que el ordenamiento de nodos priorice grado decreciente y peso."""
    G = nx.Graph()
    G.add_edge("A", "B", weight=10.0)
    G.add_edge("A", "C", weight=5.0)
    G.add_edge("B", "C", weight=20.0)
    G.add_edge("C", "D", weight=30.0)
    G.add_node("Z")  # Aislado

    # Grados: C=3, A=2, B=2, D=1, Z=0
    # Desempate entre A y B (ambos grado 2):
    #   Peso de B: 10 + 20 = 30
    #   Peso de A: 10 + 5 = 15 -> B va antes que A
    ordering = get_node_ordering(G)
    assert ordering[0] == "C"
    assert ordering[1] == "B"
    assert ordering[2] == "A"
    assert ordering[3] == "D"
    assert ordering[4] == "Z"


def test_group_by_turno_and_apply_to_graph():
    """Verifica agrupación por turno y mutación de atributos en nodos del grafo."""
    G = nx.path_graph(3)  # 0-1-2 (bipartito, colores 0, 1, 0)
    coloring = greedy_coloring(G)

    apply_coloring_to_graph(G, coloring)
    for n in G.nodes():
        assert G.nodes[n]["turno"] == coloring[n]

    turnos = group_by_turno(coloring)
    total_materias = sum(len(materias) for materias in turnos.values())
    assert total_materias == 3
