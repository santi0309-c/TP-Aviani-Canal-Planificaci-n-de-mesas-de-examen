"""
schemas.py — Modelos Pydantic de entrada y salida para la API (Tickets T17 y T18).

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
# Schemas para /cronograma (T18)
# ---------------------------------------------------------------------------

class CronogramaRequest(BaseModel):
    """Cuerpo de la petición para generar el cronograma de exámenes."""

    model_config = ConfigDict(from_attributes=True)

    umbral: float = Field(
        default=15.0,
        ge=0.0,
        le=500.0,
        description=(
            "Umbral mínimo de estudiantes afectados para considerar conflicto y crear una arista. "
            "Un umbral más bajo genera más restricciones (más turnos, menor conflicto). "
            "Un umbral más alto reduce turnos requeridos."
        ),
        examples=[15.0],
    )


class NodoCronograma(BaseModel):
    """Materia con su turno de examen asignado por el algoritmo de coloreo."""

    model_config = ConfigDict(from_attributes=True)

    id: str = Field(..., description="Identificador institucional de la materia", examples=["M01"])
    nombre: str = Field(..., description="Nombre completo de la asignatura", examples=["Álgebra y Geometría Analítica"])
    turno: int = Field(..., ge=0, description="Turno de examen asignado (entero ≥ 0)", examples=[0])
    carreras: Optional[List[str]] = Field(default=None, description="Carreras a las que pertenece", examples=[["C1", "C2", "C3"]])
    anio: Optional[int] = Field(default=None, ge=1, le=5, description="Año curricular", examples=[1])


class AristaCronograma(BaseModel):
    """Arista de conflicto activa entre dos materias."""

    model_config = ConfigDict(from_attributes=True)

    source: str = Field(..., description="Código de la materia origen", examples=["M01"])
    target: str = Field(..., description="Código de la materia destino", examples=["M02"])
    peso: float = Field(..., ge=0.0, description="Peso de la arista (estudiantes afectados estimados)", examples=[102.79])


class CronogramaResponse(BaseModel):
    """Respuesta completa del cronograma generado a partir de la integración ML + Grafo + Coloreo."""

    model_config = ConfigDict(from_attributes=True)

    umbral: float = Field(..., ge=0.0, description="Umbral utilizado para filtrar los conflictos", examples=[15.0])
    cantidad_turnos: int = Field(..., ge=0, description="Cantidad total de turnos de examen necesarios", examples=[13])
    cantidad_nodos: int = Field(..., ge=0, description="Cantidad total de materias planificadas", examples=[40])
    cantidad_aristas: int = Field(..., ge=0, description="Cantidad de aristas de conflicto activas", examples=[301])
    nodos: List[NodoCronograma] = Field(..., description="Lista de materias con su turno asignado")
    aristas: List[AristaCronograma] = Field(..., description="Lista de aristas de conflicto en el grafo")
    turnos: Optional[Dict[int, List[str]]] = Field(
        default=None,
        description="Agrupamiento de materias por turno para renderizado directo en calendario",
        examples=[{0: ["M01", "M05"], 1: ["M02", "M08"]}],
    )
    es_valido: bool = Field(
        ...,
        description="Indica si el coloreo es matemáticamente válido (sin aristas monocromáticas)",
        examples=[True],
    )
    es_coloreo_valido: Optional[bool] = Field(
        default=None,
        description="Alias de compatibilidad con validadores externos",
        examples=[True],
    )


# ---------------------------------------------------------------------------
# Schemas para /metricas (T19)
# ---------------------------------------------------------------------------

class MetricasMetadata(BaseModel):
    """Metadatos del modelo y del conjunto de test usado en la evaluación."""

    modelo: str
    semilla: int = Field(..., ge=0)
    n_test: int = Field(..., ge=0)


class MetricasRegresion(BaseModel):
    """Métricas de evaluación de la regresión."""

    mae: float = Field(..., ge=0.0)
    rmse: float = Field(..., ge=0.0)
    r2: float


class MetricasClasificacion(BaseModel):
    """Métricas de clasificación derivadas del umbral de conflicto."""

    umbral: float = Field(..., ge=0.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)


class MatrizConfusion(BaseModel):
    """Conteos de la matriz de confusión sobre el conjunto de test."""

    verdaderos_negativos: int = Field(..., ge=0)
    falsos_positivos: int = Field(..., ge=0)
    falsos_negativos: int = Field(..., ge=0)
    verdaderos_positivos: int = Field(..., ge=0)
    total: int = Field(..., ge=0)


class MetricasResponse(BaseModel):
    """Respuesta completa del endpoint GET /metricas."""

    model_config = ConfigDict(from_attributes=True)

    metadatos: MetricasMetadata
    regresion: MetricasRegresion
    clasificacion: MetricasClasificacion
    matriz_confusion: MatrizConfusion
