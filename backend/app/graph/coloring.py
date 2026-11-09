"""
coloring.py — Coloreo greedy de grafos para armar los turnos de examen.

Implementamos Welsh-Powell:
1. Ordenamos las materias según la cantidad de materias con las que tienen conflicto (grado).
2. Si empatan en grado, desempatamos por la suma de estudiantes afectados (peso de las aristas).
3. A cada materia le asignamos el primer turno (color 0, 1, 2...) que no esté ocupado por sus vecinas.
"""

from typing import Any, Dict, List, Set, Tuple
import networkx as nx


def get_node_ordering(graph: nx.Graph) -> List[Any]:
    """
    Ordena las materias para el coloreo:
    - Primero las que tienen más conflictos (mayor grado).
    - Si empatan, las que afectan a más estudiantes en total (suma de pesos).
    - Si siguen empatando, por nombre para que el resultado sea siempre el mismo.
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
    Asigna un turno (número entero >= 0) a cada materia del grafo.
    Dos materias con conflicto (conectadas por una arista) nunca reciben el mismo turno.
    """
    if graph.number_of_nodes() == 0:
        return {}

    # Ordenamos las materias para procesar primero las más conflictivas
    ordered_nodes = get_node_ordering(graph)

    coloring: Dict[Any, int] = {}

    # Le damos a cada materia el menor número de turno disponible
    for node in ordered_nodes:
        used_colors: Set[int] = {
            coloring[neighbor]
            for neighbor in graph.neighbors(node)
            if neighbor in coloring
        }

        color = 0
        while color in used_colors:
            color += 1

        coloring[node] = color

    return coloring


def validate_coloring(graph: nx.Graph, coloring: Dict[Any, int]) -> bool:
    """
    Chequea que el coloreo sea correcto:
    - Todas las materias tienen turno asignado.
    - Ninguna arista conecta dos materias con el mismo turno.
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
    """Devuelve la cantidad de turnos distintos usados."""
    if not coloring:
        return 0
    return max(coloring.values()) + 1


def group_by_turno(coloring: Dict[Any, int]) -> Dict[int, List[Any]]:
    """Agrupa las materias por turno: {0: [m1, m2], 1: [m3], ...}"""
    turnos: Dict[int, List[Any]] = {}
    for node, color in coloring.items():
        turnos.setdefault(color, []).append(node)
    return turnos


def apply_coloring_to_graph(graph: nx.Graph, coloring: Dict[Any, int]) -> None:
    """Inyecta el turno asignado como atributo 'turno' en cada nodo del grafo."""
    for node, color in coloring.items():
        if node in graph:
            graph.nodes[node]["turno"] = color
