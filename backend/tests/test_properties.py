"""
test_properties.py — Suite de pruebas automatizadas de propiedades e invariantes (Ticket T26).

Verifica formalmente las propiedades matemáticas, algorítmicas y de contrato de la API:
1. Coloreo válido y propiedades sobre grafos aleatorios sintéticos (Erdős–Rényi, Barabási–Albert, Watts–Strogatz).
2. Monotonía estricta e inclusión anidada del conjunto de aristas bajo variación del umbral τ.
3. Casos borde topológicos (grafos vacíos, independientes, completos, bipartitos, estrellas, componentes disjuntas).
4. Contrato e integración del endpoint POST /cronograma con FastAPI TestClient (esquemas, relaciones y validaciones 422).
"""

from itertools import combinations
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.main import app, clear_api_cache
from app.graph.build import (
    build_conflict_graph,
    load_materias_catalog,
    load_mock_predictions,
)
from app.graph.coloring import (
    greedy_coloring,
    validate_coloring,
    get_num_turnos,
    group_by_turno,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_cache():
    """Limpia las cachés en memoria antes y después de cada prueba."""
    clear_api_cache()
    yield
    clear_api_cache()


# ===========================================================================
# 1. Propiedades de Coloreo sobre Grafos Aleatorios (Property-Based Testing)
# ===========================================================================

class TestRandomGraphColoringProperties:
    """Verifica que el algoritmo greedy Welsh-Powell preserve los invariantes sobre familias de grafos aleatorios."""

    @pytest.mark.parametrize(
        ("n", "p", "seed"),
        [
            (10, 0.1, 42),
            (15, 0.25, 101),
            (25, 0.4, 202),
            (35, 0.15, 303),
            (40, 0.5, 404),
            (50, 0.3, 505),
        ],
    )
    def test_erdos_renyi_graphs_satisfy_coloring_invariants(self, n: int, p: float, seed: int):
        """
        En grafos binomiales/Erdős–Rényi con distintas densidades:
        - El coloreo debe ser matemáticamente válido (sin aristas monocromáticas).
        - Todo nodo debe tener un turno asignado (entero >= 0).
        - Debe respetarse la cota superior teórica de Brooks: turnos <= Delta(G) + 1.
        """
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        coloring = greedy_coloring(G)

        # 1. Cobertura completa de nodos
        assert set(coloring.keys()) == set(G.nodes)
        assert all(isinstance(turno, int) and turno >= 0 for turno in coloring.values())

        # 2. Factibilidad estricta mediante validador formal
        assert validate_coloring(G, coloring) is True

        # 3. Invariante de no adyacencia directa
        for u, v in G.edges:
            assert coloring[u] != coloring[v], f"Colisión detectada entre {u} y {v} en turno {coloring[u]}"

        # 4. Cota teórica de Brooks: Chi(G) <= Delta(G) + 1
        num_turnos = get_num_turnos(coloring)
        if n > 0 and G.number_of_edges() > 0:
            max_degree = max(dict(G.degree()).values())
            assert num_turnos <= max_degree + 1

    @pytest.mark.parametrize(
        ("n", "m", "seed"),
        [
            (15, 1, 10),
            (25, 2, 20),
            (35, 3, 30),
            (45, 2, 40),
        ],
    )
    def test_barabasi_albert_scale_free_graphs_satisfy_coloring_invariants(self, n: int, m: int, seed: int):
        """
        En grafos libres de escala Barabási–Albert (con hubs de alto grado):
        - El algoritmo debe manejar la heterogeneidad de grados sin colisiones.
        - Se debe cumplir la cota de grado máximo.
        """
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        coloring = greedy_coloring(G)

        assert validate_coloring(G, coloring) is True
        assert set(coloring.keys()) == set(G.nodes)

        for u, v in G.edges:
            assert coloring[u] != coloring[v]

        max_degree = max(dict(G.degree()).values())
        assert get_num_turnos(coloring) <= max_degree + 1

    @pytest.mark.parametrize(
        ("n", "k", "p", "seed"),
        [
            (20, 4, 0.1, 7),
            (30, 4, 0.25, 14),
            (40, 6, 0.2, 21),
        ],
    )
    def test_watts_strogatz_small_world_graphs_satisfy_coloring_invariants(
        self, n: int, k: int, p: float, seed: int
    ):
        """En grafos de mundo pequeño Watts–Strogatz (alto clustering y caminos cortos)."""
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        coloring = greedy_coloring(G)

        assert validate_coloring(G, coloring) is True
        assert set(coloring.keys()) == set(G.nodes)
        for u, v in G.edges:
            assert coloring[u] != coloring[v]


# ===========================================================================
# 2. Invariante de Monotonía del Umbral de Corte (τ)
# ===========================================================================

class TestThresholdMonotonicityInvariants:
    """Verifica formalmente las propiedades de orden e inclusión al variar el umbral."""

    def test_edge_set_nested_monotonic_inclusion(self):
        """
        PROPIEDAD DE INCLUSIÓN ESTRICTA:
        Sean tau_1 < tau_2 < ... < tau_k.
        El conjunto de aristas debe estar monótonamente anidado: E(tau_{i+1}) ⊆ E(tau_i).
        Toda arista que existe con un umbral alto DEBE estar presente con umbrales menores.
        """
        mock_preds = load_mock_predictions()
        materias = [f"M{i:02d}" for i in range(1, 41)]
        umbrales = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 150.0]

        edge_sets = []
        for tau in umbrales:
            G = build_conflict_graph(mock_preds, umbral=tau, materias=materias)
            edges_canonical = {frozenset((u, v)) for u, v in G.edges}
            edge_sets.append(edges_canonical)

        for i in range(len(umbrales) - 1):
            tau_menor = umbrales[i]
            tau_mayor = umbrales[i + 1]
            edges_menor = edge_sets[i]
            edges_mayor = edge_sets[i + 1]

            # Inclusión de subconjunto
            assert edges_mayor.issubset(edges_menor), (
                f"Violación de inclusión: aristas en tau={tau_mayor} no están en tau={tau_menor}"
            )
            # Monotonía de cardinalidad
            assert len(edges_mayor) <= len(edges_menor), (
                f"Violación de cardinalidad: |E({tau_mayor})|={len(edges_mayor)} > |E({tau_menor})|={len(edges_menor)}"
            )

    def test_node_set_strict_invariance_under_varying_threshold(self):
        """
        INVARIANTE DE CONSERVACIÓN DE MATERIAS:
        La cantidad y conjunto exacto de vértices V(tau) nunca debe alterarse al mover tau.
        Las materias que pierden aristas quedan como vértices aislados de grado 0.
        """
        mock_preds = load_mock_predictions()
        catalogo = load_materias_catalog()
        esperado_ids = {m["id"] if isinstance(m, dict) else m for m in catalogo}

        for tau in [0.0, 5.0, 15.0, 45.0, 100.0, 500.0]:
            G = build_conflict_graph(mock_preds, umbral=tau, materias=catalogo)
            assert set(G.nodes) == esperado_ids
            assert G.number_of_nodes() == len(esperado_ids)

    def test_synthetic_arbitrary_weights_monotonicity(self):
        """Verifica monotonía con una secuencia continua arbitraria de pesos sintéticos."""
        materias = ["A", "B", "C", "D", "E"]
        preds = [
            {"materia_1": "A", "materia_2": "B", "conflicto_predicho": 10.0},
            {"materia_1": "B", "materia_2": "C", "conflicto_predicho": 20.0},
            {"materia_1": "C", "materia_2": "D", "conflicto_predicho": 30.0},
            {"materia_1": "D", "materia_2": "E", "conflicto_predicho": 40.0},
            {"materia_1": "A", "materia_2": "E", "conflicto_predicho": 50.0},
        ]

        # Con tau=10.0: 5 aristas
        g10 = build_conflict_graph(preds, umbral=10.0, materias=materias)
        assert g10.number_of_edges() == 5

        # Con tau=25.0: solo aristas >= 25 (30, 40, 50 -> 3 aristas)
        g25 = build_conflict_graph(preds, umbral=25.0, materias=materias)
        assert g25.number_of_edges() == 3
        assert {frozenset(e) for e in g25.edges}.issubset({frozenset(e) for e in g10.edges})

        # Con tau=45.0: solo arista con peso 50 (1 arista)
        g45 = build_conflict_graph(preds, umbral=45.0, materias=materias)
        assert g45.number_of_edges() == 1
        assert {frozenset(e) for e in g45.edges}.issubset({frozenset(e) for e in g25.edges})

        # Con tau=55.0: 0 aristas
        g55 = build_conflict_graph(preds, umbral=55.0, materias=materias)
        assert g55.number_of_edges() == 0


# ===========================================================================
# 3. Casos Borde y Topologías Extremas (Ticket T16)
# ===========================================================================

class TestTopologicalEdgeCases:
    """Valida exhaustivamente casos límite de la teoría de grafos y asignación de exámenes."""

    def test_completely_empty_graph(self):
        """Grafo sin nodos ni aristas."""
        G = nx.Graph()
        coloring = greedy_coloring(G)
        assert coloring == {}
        assert get_num_turnos(coloring) == 0
        assert validate_coloring(G, coloring) is True

    def test_edgeless_independent_set_collapses_to_single_turn(self):
        """Un conjunto totalmente independiente (sin conflictos) debe ubicarse en el turno 0."""
        materias = [f"M{i:02d}" for i in range(1, 21)]
        G = build_conflict_graph([], umbral=15.0, materias=materias)

        coloring = greedy_coloring(G)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() == 0
        assert get_num_turnos(coloring) == 1
        assert all(t == 0 for t in coloring.values())
        assert validate_coloring(G, coloring) is True

    def test_complete_graph_clique_requires_exact_node_count_turns(self):
        """En un grafo completo K_n, cada nodo requiere un turno distinto (n turnos)."""
        n = 8
        G = nx.complete_graph(n)
        coloring = greedy_coloring(G)

        assert get_num_turnos(coloring) == n
        assert len(set(coloring.values())) == n
        assert validate_coloring(G, coloring) is True

    def test_complete_bipartite_graph_bounded_by_two_turns(self):
        """En un grafo bipartito completo K_{r,s}, el coloreo debe requerir a lo sumo 2 turnos."""
        G = nx.complete_bipartite_graph(5, 7)
        coloring = greedy_coloring(G)

        assert get_num_turnos(coloring) <= 2
        assert validate_coloring(G, coloring) is True

    def test_star_graph_requires_exactly_two_turns(self):
        """Grafo estrella: 1 nodo central conectado a k hojas independientes."""
        G = nx.star_graph(12)  # 1 centro + 12 hojas = 13 nodos
        coloring = greedy_coloring(G)

        assert get_num_turnos(coloring) == 2
        assert validate_coloring(G, coloring) is True

    def test_disconnected_cliques_efficiently_reuse_turnos(self):
        """
        REUTILIZACIÓN ÓPTIMA ENTRE COMPONENTES CONEXAS:
        Dos cliques disjuntas K_4 y K_4 sin aristas intermedias deben requerir 4 turnos, no 8.
        """
        G = nx.Graph()
        # Componente 1: materias A1..A4
        for u, v in combinations(["A1", "A2", "A3", "A4"], 2):
            G.add_edge(u, v)
        # Componente 2: materias B1..B4
        for u, v in combinations(["B1", "B2", "B3", "B4"], 2):
            G.add_edge(u, v)

        assert nx.number_connected_components(G) == 2
        coloring = greedy_coloring(G)

        assert get_num_turnos(coloring) == 4
        assert validate_coloring(G, coloring) is True

    def test_extreme_threshold_boundaries(self):
        """
        Comportamiento en los límites de umbral:
        - tau <= 0.0: incluye todo par con predicción >= 0.
        - tau -> inf: ninguna arista creada, todas las materias en 1 solo turno.
        """
        mock_preds = load_mock_predictions()
        catalogo = load_materias_catalog()

        # Umbral cero: máxima cantidad de aristas
        g_cero = build_conflict_graph(mock_preds, umbral=0.0, materias=catalogo)
        assert g_cero.number_of_edges() == len(mock_preds)

        # Umbral infinito (mayor al maximo de conflicto predicho)
        g_inf = build_conflict_graph(mock_preds, umbral=1000.0, materias=catalogo)
        assert g_inf.number_of_edges() == 0
        coloring_inf = greedy_coloring(g_inf)
        assert get_num_turnos(coloring_inf) == 1


# ===========================================================================
# 4. Integración del Endpoint POST /cronograma con TestClient
# ===========================================================================

class TestEndpointCronogramaIntegrationContract:
    """Verifica el contrato de la API, estructura de respuesta y validaciones de FastAPI."""

    @pytest.mark.parametrize("umbral", [0.0, 10.0, 15.0, 35.0, 80.0, 200.0])
    def test_post_cronograma_successful_schema_and_types(self, umbral: float):
        """Peticiones válidas deben responder HTTP 200 con tipos y esquema completos."""
        response = client.post("/cronograma", json={"umbral": umbral})
        assert response.status_code == 200

        data = response.json()
        assert data["umbral"] == umbral
        assert isinstance(data["cantidad_nodos"], int)
        assert isinstance(data["cantidad_aristas"], int)
        assert isinstance(data["cantidad_turnos"], int)
        assert isinstance(data["es_valido"], bool)
        assert data["es_valido"] is True

        assert isinstance(data["nodos"], list)
        assert isinstance(data["aristas"], list)
        assert isinstance(data["turnos"], dict)

        # Verificación de formato de cada nodo
        for nodo in data["nodos"]:
            assert "id" in nodo
            assert "nombre" in nodo
            assert "turno" in nodo
            assert isinstance(nodo["turno"], int) and nodo["turno"] >= 0

        # Verificación de formato de cada arista
        for arista in data["aristas"]:
            assert "source" in arista
            assert "target" in arista
            assert "peso" in arista
            assert isinstance(arista["peso"], (int, float))
            assert arista["peso"] >= umbral

    def test_post_cronograma_relational_integrity(self):
        """
        INTEGRIDAD RELACIONAL:
        La suma de materias agrupadas en el diccionario 'turnos' debe coincidir exactamente
        con 'cantidad_nodos' y con la longitud de la lista 'nodos'.
        """
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 200

        data = response.json()
        total_en_turnos = sum(len(materias) for materias in data["turnos"].values())

        assert total_en_turnos == data["cantidad_nodos"]
        assert total_en_turnos == len(data["nodos"])

    def test_post_cronograma_academic_non_adjacency_rule(self):
        """
        REGLA ACADÉMICA INVIOLABLE:
        Ningún par de materias conectado por una arista de conflicto devuelta por la API
        puede tener asignado el mismo turno de examen.
        """
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 200

        data = response.json()
        turnos_map = {nodo["id"]: nodo["turno"] for nodo in data["nodos"]}

        for arista in data["aristas"]:
            src = arista["source"]
            tgt = arista["target"]
            assert turnos_map[src] != turnos_map[tgt], (
                f"Conflicto de examen: materias adyacentes {src} y {tgt} comparten turno {turnos_map[src]}"
            )

    @pytest.mark.parametrize(
        ("payload", "expected_status"),
        [
            ({"umbral": -1.0}, 422),     # Umbral menor al mínimo ge=0.0
            ({"umbral": -50.0}, 422),    # Umbral negativo
            ({"umbral": 600.0}, 422),    # Umbral superior al límite le=500.0
            ({"umbral": "quince"}, 422), # Tipo inválido no numérico
            ({"umbral": None}, 422),     # Tipo None no permitido
        ],
    )
    def test_post_cronograma_validation_errors(self, payload: dict, expected_status: int):
        """Peticiones que violan las restricciones de Pydantic deben retornar HTTP 422 Unprocessable Entity."""
        response = client.post("/cronograma", json=payload)
        assert response.status_code == expected_status
        error_detail = response.json()
        assert "detail" in error_detail

