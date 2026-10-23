"""build.py — Construcción del grafo de conflictos a partir de predicciones."""
# TODO: crear grafo con networkx; agregar arista si conflicto predicho >= umbral


def build_conflict_graph(predictions, umbral: float):
    """
    predictions: lista de dicts {materia_1, materia_2, conflicto_predicho}
    umbral: float — umbral mínimo para agregar arista

    Retorna un networkx.Graph con pesos en las aristas.
    """
    raise NotImplementedError
