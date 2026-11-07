"""
main.py — Servidor y API REST con FastAPI (Tickets T02 y T17).

Para ejecutar localmente:
    uvicorn app.main:app --reload --port 8000
Documentación Swagger interactiva:
    http://localhost:8000/docs
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ConflictosResponse, MateriaSchema

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
    version="0.2.0",
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


@app.get("/", tags=["utils"])
def root():
    """Endpoint raíz con mensaje de bienvenida y estado de la API."""
    return {"status": "ok", "mensaje": "API de planificación de mesas de examen"}


@app.get("/health", tags=["utils"])
def health_check():
    """Endpoint de salud para monitoreo y verificación de operatividad."""
    return {"status": "ok", "mensaje": "el servidor está vivo"}


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
