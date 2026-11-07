"""
coloring.py — Coloreo greedy del grafo de conflictos (Ticket T14).

Implementación propia del algoritmo voraz (Welsh-Powell) para asignación de turnos de examen.
NetworkX solo se utiliza como estructura de datos para representar el grafo.
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx


def get_node_ordering(graph: nx.Graph) -> List[Any]:
    """
    Ordena los nodos del grafo aplicando la heurística de Welsh-Powell extendida:
    1. Criterio primario: Grado decreciente (mayor cantidad de vecinos/conflictos).
    2. Criterio secundario: Suma de pesos de aristas incidentes decreciente (mayor gravedad).
    3. Criterio terciario: Identificador del nodo (orden lexicográfico para determinismo).

    Complejidad: O(|V| log |V|).
    """
    def sort_key(node: Any) -> Tuple[int, float, str]:
        degree = graph.degree(node)
        weight_sum = sum(
            float(data.get("weight", 0.0))
            for _, _, data in graph.edges(node, data=True)
        )
        return (-degree, -weight_sum, str(node))

    return sorted(graph.nodes(), key=sort_key)


def greedy_coloring(graph: nx.Graph) -> Dict[Any, int]:
    """
    Aplica coloreo greedy (Welsh-Powell) sobre el grafo de conflictos.

    Parámetros:
        graph (nx.Graph): Grafo no dirigido con las materias como nodos.

    Retorna:
        Dict[Any, int]: Mapeo {materia: turno} donde turno es un entero >= 0.

    Garantía teórica:
        Utiliza a lo sumo Delta(G) + 1 colores (donde Delta es el grado máximo del grafo).

    Complejidad:
        O(|V| log |V| + |E|) en tiempo, O(|V|) en espacio.
    """
    if graph.number_of_nodes() == 0:
        return {}

    # 1. Obtener orden de los nodos por grado y gravedad de conflicto
    ordered_nodes = get_node_ordering(graph)

    coloring: Dict[Any, int] = {}

    # 2. Asignar vorazmente el menor color disponible a cada nodo
    for node in ordered_nodes:
        # Colores usados por vecinos que ya fueron coloreados
        used_colors: Set[int] = {
            coloring[neighbor]
            for neighbor in graph.neighbors(node)
            if neighbor in coloring
        }

        # Menor entero >= 0 que no esté en used_colors
        color = 0
        while color in used_colors:
            color += 1

        coloring[node] = color

    return coloring


def validate_coloring(graph: nx.Graph, coloring: Dict[Any, int]) -> bool:
    """
    Verifica que el coloreo sea matemáticamente válido:
    1. Todas las materias del grafo deben tener un color asignado (entero >= 0).
    2. Ninguna arista puede tener extremos con el mismo color.

    Parámetros:
        graph (nx.Graph): Grafo de conflictos.
        coloring (Dict[Any, int]): Mapeo {materia: turno}.

    Retorna:
        bool: True si el coloreo es válido, False en caso de conflicto.
    """
    # Verificar que todos los nodos tengan un color no negativo asignado
    for node in graph.nodes():
        if node not in coloring:
            return False
        if not isinstance(coloring[node], int) or coloring[node] < 0:
            return False

    # Verificar que ningún par adyacente comparta color
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            return False

    return True


def get_num_turnos(coloring: Dict[Any, int]) -> int:
    """Retorna la cantidad total de turnos (colores) utilizados."""
    if not coloring:
        return 0
    return max(coloring.values()) + 1


def group_by_turno(coloring: Dict[Any, int]) -> Dict[int, List[Any]]:
    """
    Agrupa las materias por turno asignado: {turno_0: [m1, m2], turno_1: [m3], ...}
    Útil para la visualización del cronograma y calendario en el frontend.
    """
    turnos: Dict[int, List[Any]] = {}
    for node, color in coloring.items():
        turnos.setdefault(color, []).append(node)
    return turnos


def apply_coloring_to_graph(graph: nx.Graph, coloring: Dict[Any, int]) -> None:
    """Inyecta el turno asignado como atributo 'turno' en cada nodo del grafo."""
    for node, color in coloring.items():
        if node in graph:
            graph.nodes[node]["turno"] = color
