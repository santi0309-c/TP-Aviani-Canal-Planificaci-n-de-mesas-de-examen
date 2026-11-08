"""
main.py — Servidor y API REST con FastAPI (Tickets T02, T17 y T18).

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
    AristaCronograma,
    ConflictoSchema,
    ConflictosResponse,
    CronogramaRequest,
    CronogramaResponse,
    MateriaSchema,
    NodoCronograma,
)

# Rutas predeterminadas robustas relativas a la ubicación del script
BASE_DIR = Path(__file__).resolve().parent      # backend/app
BACKEND_DIR = BASE_DIR.parent                   # backend
MATERIAS_JSON_PATH = BACKEND_DIR / "data" / "materias.json"

# Variables para almacenamiento en caché en memoria
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


# Instancia principal de la aplicación FastAPI
app = FastAPI(
    title="Planificación de Mesas de Examen",
    description=(
        "API para optimización de cronogramas de exámenes universitarios "
        "mediante modelos de Machine Learning y algoritmos de coloreo de grafos. "
        "Desarrollado para el TP — Aviani & Canal."
    ),
    version="0.3.0",
)

# Configuración de CORS para integración fluida con Vite / React
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
# Endpoints de lectura (T17)
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
# Endpoint de integración: Generación de Cronograma (T18)
# ---------------------------------------------------------------------------

@app.post(
    "/cronograma",
    response_model=CronogramaResponse,
    tags=["cronograma"],
    summary="Construir grafo y generar cronograma de exámenes",
    description=(
        "Endpoint integrador central del proyecto. Orquesta el pipeline completo: "
        "(1) Infiere los conflictos continuos mediante el modelo de regresión entrenado, "
        "(2) Construye el grafo no dirigido filtrando aristas con conflicto predicho ≥ umbral τ, "
        "(3) Aplica el algoritmo voraz Welsh-Powell para asignar turnos disjuntos a materias adyacentes, "
        "(4) Retorna el cronograma validado junto a los nodos coloreados y aristas de conflicto."
    ),
)
def post_cronograma(request: CronogramaRequest) -> CronogramaResponse:
    """
    Genera el cronograma de exámenes óptimo para un umbral dado.

    El umbral τ modula el compromiso (trade-off) del problema:
    - Umbral bajo: Más aristas de conflicto → más turnos requeridos, menor riesgo de superposición.
    - Umbral alto: Menos aristas → menor cantidad de turnos requeridos, posibles conflictos menores.
    """
    from app.graph.build import build_conflict_graph
    from app.graph.coloring import (
        get_num_turnos,
        greedy_coloring,
        group_by_turno,
        validate_coloring,
    )

    umbral = request.umbral

    # 1. Obtener predicciones cacheadas y catálogo de materias
    try:
        predicciones = get_cached_predictions()
        materias_catalog = get_materias_catalog()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="No se pudo generar el cronograma: no se encontró el modelo o los datos requeridos.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error inesperado al cargar datos de inferencia: {str(exc)}",
        ) from exc

    # 2. Construir el grafo de conflictos (asegurando materias aisladas)
    G = build_conflict_graph(
        predictions=predicciones,
        umbral=umbral,
        materias=materias_catalog,
    )

    # 3. Aplicar coloreo greedy con heurística Welsh-Powell
    coloring = greedy_coloring(G)

    # 4. Validar factibilidad formal del coloreo
    es_valido = bool(validate_coloring(G, coloring))

    # 5. Métricas y agrupaciones
    num_turnos = get_num_turnos(coloring)
    turnos_agrupados = group_by_turno(coloring)

    # Mapeo de metadatos de materias {id -> dict}
    materias_dict: Dict[str, Dict[str, Any]] = {
        m["id"]: m
        for m in materias_catalog
        if isinstance(m, dict) and "id" in m
    }

    # 6. Serializar nodos con turnos y atributos académicos
    nodos = [
        {
            "id": str(node),
            "nombre": materias_dict.get(str(node), {}).get("nombre", str(node)),
            "turno": coloring.get(node, 0),
            "carreras": materias_dict.get(str(node), {}).get("carreras"),
            "anio": materias_dict.get(str(node), {}).get("anio"),
        }
        for node in G.nodes()
    ]

    # 7. Serializar aristas de conflicto activas
    aristas = [
        {
            "source": str(u),
            "target": str(v),
            "peso": float(data.get("weight", data.get("conflicto", 0.0))),
        }
        for u, v, data in G.edges(data=True)
    ]

    # 8. Agrupamiento para consumo directo de frontend / vista de calendario
    turnos_response: Dict[int, List[str]] = {
        int(turno): [str(m) for m in materias]
        for turno, materias in turnos_agrupados.items()
    }

    return CronogramaResponse(
        umbral=umbral,
        cantidad_turnos=num_turnos,
        cantidad_nodos=G.number_of_nodes(),
        cantidad_aristas=G.number_of_edges(),
        nodos=nodos,
        aristas=aristas,
        turnos=turnos_response,
        es_valido=es_valido,
        es_coloreo_valido=es_valido,
    )
