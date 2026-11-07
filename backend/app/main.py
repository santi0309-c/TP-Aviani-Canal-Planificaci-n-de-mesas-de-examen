"""
main.py — Servidor y API REST con FastAPI (Tickets T02, T15, T17).

Para ejecutar localmente:
    uvicorn app.main:app --reload --port 8000
Documentación Swagger interactiva:
    http://localhost:8000/docs
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    ConflictosResponse,
    CronogramaRequest,
    CronogramaResponse,
    MateriaSchema,
)

# Rutas predeterminadas robustas relativas a la ubicación del script
BASE_DIR = Path(__file__).resolve().parent      # backend/app
BACKEND_DIR = BASE_DIR.parent                   # backend
MATERIAS_JSON_PATH = BACKEND_DIR / "data" / "materias.json"

# Cachés en memoria: evita releer disco o reinferir en cada request
_MATERIAS_CACHE: Optional[List[Dict[str, Any]]] = None
_PREDICCIONES_CACHE: Optional[List[Dict[str, Any]]] = None


def clear_api_cache() -> None:
    """Limpia las cachés en memoria de la API (útil para pruebas unitarias)."""
    global _MATERIAS_CACHE, _PREDICCIONES_CACHE
    _MATERIAS_CACHE = None
    _PREDICCIONES_CACHE = None


def get_materias_catalog() -> List[Dict[str, Any]]:
    """Carga y almacena en memoria el catálogo de materias desde materias.json."""
    global _MATERIAS_CACHE
    if _MATERIAS_CACHE is None:
        if not MATERIAS_JSON_PATH.exists():
            raise FileNotFoundError(f"No se encontró el catálogo de materias en: {MATERIAS_JSON_PATH}")
        with open(MATERIAS_JSON_PATH, "r", encoding="utf-8") as f:
            _MATERIAS_CACHE = json.load(f)
    return _MATERIAS_CACHE


def get_cached_predictions() -> List[Dict[str, Any]]:
    """Carga y almacena en memoria las predicciones de conflictos para todos los pares."""
    global _PREDICCIONES_CACHE
    if _PREDICCIONES_CACHE is None:
        from app.ml.predict import predecir_pares
        _PREDICCIONES_CACHE = predecir_pares(as_dict=True)
    return _PREDICCIONES_CACHE


# ---------------------------------------------------------------------------
# Instancia principal de FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Planificación de Mesas de Examen",
    description=(
        "API para optimización de cronogramas de exámenes universitarios "
        "mediante modelos de Machine Learning y algoritmos de coloreo de grafos. "
        "Desarrollado para el TP — Aviani & Canal."
    ),
    version="0.3.0",
)

# CORS: 5173 es el puerto default de Vite
# sin esto el browser bloquea todo y te volvés loco debuggeando xd
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints de utilidades
# ---------------------------------------------------------------------------

@app.get("/", tags=["utils"])
def root():
    """Endpoint raíz con mensaje de bienvenida y estado de la API."""
    return {"status": "ok", "mensaje": "API de planificación de mesas de examen"}


@app.get("/health", tags=["utils"])
def health_check():
    """Endpoint de salud para monitoreo y verificación de operatividad."""
    return {"status": "ok", "mensaje": "el servidor está vivo"}


# ---------------------------------------------------------------------------
# Endpoint GET /materias (T17)
# ---------------------------------------------------------------------------

@app.get(
    "/materias",
    response_model=List[MateriaSchema],
    tags=["materias"],
    summary="Obtener catálogo de materias",
    description="Devuelve el listado completo de las 40 materias registradas en el sistema con sus datos académicos.",
)
def get_materias() -> List[Dict[str, Any]]:
    """Devuelve el catálogo completo de materias."""
    return get_materias_catalog()


# ---------------------------------------------------------------------------
# Endpoint GET /conflictos (T17)
# ---------------------------------------------------------------------------

@app.get(
    "/conflictos",
    response_model=ConflictosResponse,
    tags=["conflictos"],
    summary="Obtener pares de materias en conflicto según umbral",
    description=(
        "Estima qué pares de materias presentan conflicto potencial superando el umbral "
        "de estudiantes afectados (τ). Utiliza el modelo predictivo de regresión entrenado."
    ),
)
def get_conflictos(
    umbral: float = Query(
        default=15.0,
        ge=0.0,
        le=500.0,
        description="Umbral mínimo de estudiantes afectados para considerar conflicto.",
        examples=[15.0],
    ),
) -> ConflictosResponse:
    """Devuelve los pares con conflicto predicho >= umbral."""
    all_preds = get_cached_predictions()
    conflictos_filtrados = [
        p for p in all_preds
        if float(p.get("conflicto_predicho", 0.0)) >= umbral
    ]
    return ConflictosResponse(
        umbral=umbral,
        total_conflictos=len(conflictos_filtrados),
        conflictos=conflictos_filtrados,
    )


# ---------------------------------------------------------------------------
# Endpoint POST /cronograma (T15)
# ---------------------------------------------------------------------------

@app.post(
    "/cronograma",
    response_model=CronogramaResponse,
    tags=["cronograma"],
    summary="Construir grafo de conflictos y generar cronograma de exámenes",
    description=(
        "Integra el pipeline completo: "
        "(1) obtiene las predicciones del modelo de ML, "
        "(2) construye el grafo de conflictos filtrando por el umbral τ, "
        "(3) aplica el algoritmo de coloreo greedy (Welsh-Powell) para asignar turnos de examen, "
        "(4) devuelve el cronograma completo con nodos, aristas y agrupamiento por turno."
    ),
)
def post_cronograma(body: CronogramaRequest) -> CronogramaResponse:
    """
    Construye el grafo de conflictos y aplica coloreo greedy para asignar turnos.

    El umbral τ controla el trade-off:
    - Umbral bajo → más aristas → más turnos, menos estudiantes perjudicados.
    - Umbral alto → menos aristas → menos turnos, posibles conflictos sin resolver.
    """
    from app.graph.build import build_conflict_graph
    from app.graph.coloring import (
        greedy_coloring,
        validate_coloring,
        get_num_turnos,
        group_by_turno,
    )

    umbral = body.umbral

    # 1. Obtener predicciones cacheadas y catálogo de materias
    try:
        predicciones = get_cached_predictions()
        materias_catalog = get_materias_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="No se pudo generar el cronograma porque falta el modelo o alguno de los datos requeridos.",
        ) from exc

    # 3. Construir el grafo de conflictos
    G = build_conflict_graph(
        predictions=predicciones,
        umbral=umbral,
        materias=materias_catalog,
    )

    # 4. Aplicar coloreo greedy (Welsh-Powell)
    coloring = greedy_coloring(G)

    # 5. Validar coloreo
    es_valido = validate_coloring(G, coloring)

    # 6. Calcular estadísticas de turnos
    num_turnos = get_num_turnos(coloring)
    turnos_agrupados = group_by_turno(coloring)

    # 7. Construir catálogo de nombres {id -> nombre}
    nombres_map: Dict[str, str] = {
        m["id"]: m.get("nombre", m["id"])
        for m in materias_catalog
        if isinstance(m, dict) and "id" in m
    }

    # 8. Serializar nodos con turno asignado
    nodos = [
        {
            "id": node,
            "nombre": nombres_map.get(str(node), str(node)),
            "turno": coloring.get(node, 0),
        }
        for node in G.nodes()
    ]

    # 9. Serializar aristas
    aristas = [
        {
            "source": u,
            "target": v,
            "peso": float(data.get("weight", 0.0)),
        }
        for u, v, data in G.edges(data=True)
    ]

    # 10. Convertir turnos agrupados a {int: List[str]}
    turnos_response: Dict[int, List[str]] = {
        turno: [str(m) for m in materias]
        for turno, materias in turnos_agrupados.items()
    }

    return CronogramaResponse(
        umbral=umbral,
        cantidad_turnos=num_turnos,
        cantidad_nodos=G.number_of_nodes(),
        cantidad_aristas=G.number_of_edges(),
        es_coloreo_valido=es_valido,
        turnos=turnos_response,
        nodos=nodos,
        aristas=aristas,
    )
