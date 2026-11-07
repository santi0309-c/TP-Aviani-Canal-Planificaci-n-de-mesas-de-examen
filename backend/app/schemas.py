"""
schemas.py — Modelos Pydantic de entrada y salida para la API (Tickets T17 y T15).

Define los contratos de datos fuertemente tipados para:
- /materias:    Catálogo de materias institucionales (nodos).
- /conflictos:  Estimación de pares de materias en conflicto según umbral (aristas).
- /cronograma:  Construcción del grafo + coloreo greedy → cronograma de turnos de examen.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Schemas compartidos de materias (nodos del grafo)
# ---------------------------------------------------------------------------

class MateriaSchema(BaseModel):
    """Esquema representativo de una materia académica (nodo del grafo)."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(
        ...,
        description="Identificador institucional único de la materia",
        examples=["M01"],
    )
    nombre: str = Field(
        ...,
        description="Nombre completo de la asignatura",
        examples=["Álgebra y Geometría Analítica"],
    )
    anio: int = Field(
        ...,
        ge=1,
        le=5,
        description="Año lectivo del plan de estudios al que pertenece (1 a 5)",
        examples=[1],
    )
    carreras: List[str] = Field(
        ...,
        description="Lista de códigos de carreras que cursan la materia",
        examples=[["C1", "C2", "C3"]],
    )
    tamanio: int = Field(
        ...,
        ge=0,
        description="Matrícula histórica o cantidad estimada de estudiantes inscriptos",
        examples=[255],
    )


# ---------------------------------------------------------------------------
# Schemas para /conflictos (T17)
# ---------------------------------------------------------------------------

class ConflictoSchema(BaseModel):
    """Esquema de un par de materias con su conflicto de exámenes estimado."""

    model_config = ConfigDict(from_attributes=True)

    materia_1: str = Field(
        ...,
        description="Código de la primera materia del par",
        examples=["M01"],
    )
    materia_2: str = Field(
        ...,
        description="Código de la segunda materia del par",
        examples=["M02"],
    )
    conflicto_predicho: float = Field(
        ...,
        ge=0.0,
        description="Cantidad estimada de estudiantes afectados en conflicto (peso potencial de la arista)",
        examples=[102.79],
    )
    cant_estudiantes_compartidos_historico: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estudiantes que cursaron o rindieron ambas materias históricamente",
        examples=[168],
    )
    mismo_anio: Optional[bool] = Field(
        default=None,
        description="Indica si ambas materias pertenecen al mismo año de la carrera",
        examples=[True],
    )
    misma_carrera: Optional[bool] = Field(
        default=None,
        description="Indica si ambas materias pertenecen a la misma carrera",
        examples=[True],
    )


class ConflictosResponse(BaseModel):
    """Respuesta del endpoint /conflictos con el conjunto de aristas filtradas por umbral."""

    model_config = ConfigDict(from_attributes=True)

    umbral: float = Field(
        ...,
        ge=0.0,
        description="Umbral de corte aplicado (cantidad mínima de estudiantes para considerar conflicto)",
        examples=[15.0],
    )
    total_conflictos: int = Field(
        ...,
        ge=0,
        description="Cantidad total de pares de materias que superan el umbral",
        examples=[301],
    )
    conflictos: List[ConflictoSchema] = Field(
        ...,
        description="Lista detallada de los pares en conflicto",
    )


# ---------------------------------------------------------------------------
# Schemas para /cronograma (T15)
# ---------------------------------------------------------------------------

class CronogramaRequest(BaseModel):
    """Cuerpo del request POST /cronograma."""

    model_config = ConfigDict(from_attributes=True)

    umbral: float = Field(
        default=15.0,
        ge=0.0,
        le=500.0,
        description=(
            "Umbral mínimo de estudiantes afectados para agregar una arista al grafo. "
            "Un umbral más bajo genera más restricciones (más turnos, menos conflictos). "
            "Un umbral más alto reduce turnos pero puede dejar conflictos sin resolver."
        ),
        examples=[15.0],
    )


class NodoCronograma(BaseModel):
    """Nodo del grafo en la respuesta de cronograma: materia con turno asignado."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Identificador de la materia", examples=["M01"])
    nombre: str = Field(..., description="Nombre de la materia", examples=["Análisis Matemático I"])
    turno: int = Field(..., ge=0, description="Turno de examen asignado (entero ≥ 0)", examples=[0])


class AristaCronograma(BaseModel):
    """Arista del grafo de conflictos tal como se devuelve en el cronograma."""

    model_config = ConfigDict(from_attributes=True)

    source: str = Field(..., description="Materia origen", examples=["M01"])
    target: str = Field(..., description="Materia destino", examples=["M07"])
    peso: float = Field(..., ge=0.0, description="Peso de la arista (estudiantes afectados estimados)", examples=[23.4])


class CronogramaResponse(BaseModel):
    """Respuesta completa del endpoint POST /cronograma."""

    model_config = ConfigDict(from_attributes=True)

    umbral: float = Field(..., ge=0.0, description="Umbral utilizado para construir el grafo", examples=[15.0])
    cantidad_turnos: int = Field(
        ..., ge=0,
        description="Número de turnos distintos necesarios (colores del coloreo greedy)",
        examples=[5],
    )
    cantidad_nodos: int = Field(..., ge=0, description="Total de materias en el grafo", examples=[40])
    cantidad_aristas: int = Field(..., ge=0, description="Total de aristas (pares en conflicto)", examples=[301])
    es_coloreo_valido: bool = Field(
        ...,
        description="True si ninguna arista tiene ambos extremos en el mismo turno",
    )
    turnos: Dict[int, List[str]] = Field(
        ...,
        description="Agrupamiento de materias por turno: {turno_id: [materia_id, ...]}",
        examples=[{0: ["M01", "M05"], 1: ["M02", "M08"]}],
    )
    nodos: List[NodoCronograma] = Field(..., description="Lista de nodos con turno asignado")
    aristas: List[AristaCronograma] = Field(..., description="Lista de aristas del grafo de conflictos")
