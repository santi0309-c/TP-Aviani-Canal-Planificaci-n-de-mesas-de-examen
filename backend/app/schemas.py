"""
schemas.py — Modelos Pydantic de entrada y salida para la API (Ticket T17).

Define los contratos de datos fuertemente tipados para:
- /materias: Catálogo de materias institucionales (nodos).
- /conflictos: Estimación de pares de materias en conflicto según umbral (aristas).
"""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


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
