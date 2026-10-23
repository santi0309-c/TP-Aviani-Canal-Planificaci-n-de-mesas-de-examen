"""coloring.py — Coloreo greedy del grafo de conflictos."""
# Implementación propia; networkx solo se usa para representar el grafo.


def greedy_coloring(graph) -> dict:
    """
    Aplica coloreo greedy (Welsh-Powell: nodos ordenados por grado decreciente).

    Retorna dict {nodo: color} donde color es un entero >= 0 (turno asignado).
    """
    raise NotImplementedError


def validate_coloring(graph, coloring: dict) -> bool:
    """Verifica que ninguna arista tenga extremos del mismo color."""
    for u, v in graph.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True
