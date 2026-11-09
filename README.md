# Planificación de Mesas de Examen mediante Grafos y Machine Learning

**Integrantes:**  
- Valentino Aviani · Legajo 51522  
- Santiago Cañal · Legajo 50875  

**Cátedra:** Inteligencia Artificial / Algoritmos — UTN FRRo  
**Entrega:** 08/11/2026  

---

## Tabla de contenidos

1. [Descripción del problema](#1-descripción-del-problema)
2. [Cómo funciona (visión general)](#2-cómo-funciona-visión-general)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Instalación y ejecución](#4-instalación-y-ejecución)
5. [Dataset](#5-dataset)
6. [Modelo predictivo](#6-modelo-predictivo)
7. [Algoritmo de grafos](#7-algoritmo-de-grafos)
8. [Integración modelo → grafo](#8-integración-modelo--grafo)
9. [API del backend](#9-api-del-backend)
10. [Frontend](#10-frontend)
11. [Tests](#11-tests)
12. [Uso de IA](#12-uso-de-ia)
13. [Limitaciones y extensiones](#13-limitaciones-y-extensiones)
14. [Fuentes y licencia](#14-fuentes-y-licencia)

---

## 1. Descripción del problema

La secretaría académica de una facultad debe armar el cronograma de mesas de examen. Si dos materias con muchos estudiantes en común se rinden el mismo día y turno, esos estudiantes tienen que elegir cuál rendir y pierden una oportunidad.

Este sistema:

1. **Estima** qué pares de materias van a tener conflicto, a partir de datos históricos.
2. **Construye** un grafo de conflictos usando esa estimación.
3. **Asigna** turnos de examen mediante **coloreo greedy** (Welsh-Powell), evitando que dos materias con conflicto queden en el mismo turno. El algoritmo no garantiza la cantidad mínima de turnos posible (eso es NP-difícil), pero en la práctica da resultados razonables.
4. **Muestra** el resultado en una aplicación web donde se puede variar el umbral de conflicto y ver cómo cambia el cronograma.

## 2. Cómo funciona (visión general)

```
dataset de pares ──► modelo (regresión lineal) ──► conflicto predicho por par
                                                            │
                                        umbral configurable ▼
                                              grafo de conflictos (aristas ponderadas)
                                                            │
                                                            ▼
                                                coloreo greedy (turnos)
                                                            │
                                                            ▼
                                        API (backend)  ──►  frontend (grafo + calendario)
```

- **Nodo:** materia que se rinde en el período.
- **Arista:** conflicto predicho entre dos materias. El **peso** es la magnitud estimada del conflicto (cantidad de estudiantes afectados).
- **Color:** turno de examen asignado.

Si se cambia el umbral, el grafo cambia y el coloreo se recalcula en tiempo real.

## 3. Estructura del repositorio

```
.
├── backend/
│   ├── app/
│   │   ├── main.py                # API FastAPI y endpoints REST
│   │   ├── schemas.py             # Modelos de datos Pydantic v2
│   │   ├── graph/
│   │   │   ├── build.py           # Construcción del grafo de conflictos (NetworkX)
│   │   │   └── coloring.py        # Algoritmo de coloreo greedy (Welsh-Powell)
│   │   └── ml/
│   │       ├── features.py        # Pipeline de preprocesamiento y partición de datos
│   │       ├── train.py           # Entrenamiento, GridSearchCV y selección de modelos
│   │       ├── predict.py         # Carga del pipeline y predicción continua
│   │       ├── evaluate.py        # Evaluación sobre test y generación de metricas.json
│   │       └── error_analysis.py  # Diagnóstico de residuos y casos borde
│   ├── data/
│   │   ├── generate_dataset.py    # Generador determinístico del dataset sintético
│   │   ├── materias.json          # Catálogo de las 40 asignaturas
│   │   └── pares_materias.csv     # Dataset de 780 pares (C(40,2))
│   ├── models/
│   │   ├── modelo.joblib          # Pipeline entrenado (Regresión Lineal)
│   │   └── metricas.json          # Métricas de test generadas por evaluate.py
│   ├── tests/                     # Suite de 166 pruebas con pytest
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Componente principal y layout general
│   │   ├── GraphVisualization.tsx # Grafo interactivo con Cytoscape.js
│   │   ├── CronogramaGrid.tsx     # Grilla y buscador de turnos de examen
│   │   ├── MetricsPanel.tsx       # Panel de métricas del cronograma y del modelo ML
│   │   ├── UmbralSelector.tsx     # Slider de umbral τ con presets y debounce
│   │   ├── api.ts                 # Cliente HTTP centralizado (fetch + AbortController)
│   │   ├── colors.ts              # Colores por turno (ángulo áureo + contraste WCAG AA)
│   │   ├── types.ts               # Tipos TypeScript compartidos
│   │   ├── useApiRequest.ts       # Hook de requests con AbortController
│   │   └── useDebounce.ts         # Hook de debounce para slider y búsqueda
│   └── package.json
├── informe/                       # Documentación teórica, análisis y capturas
└── README.md
```

## 4. Instalación y ejecución

### Requisitos

- Python 3.11+
- Node.js 20+ y npm
- Git

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Generar el dataset (reproducible por semilla)
python data/generate_dataset.py --seed 42 --n-materias 40 --out data/pares_materias.csv

# 2) Entrenar y guardar el modelo
python -m app.ml.train --data data/pares_materias.csv --out models/modelo.joblib

# 3) Evaluar el modelo sobre el conjunto de test y generar metricas.json
#    (sin este archivo el endpoint /metricas devuelve 503 y el panel de ML no carga)
python -m app.ml.evaluate --data data/pares_materias.csv --model models/modelo.joblib

# 4) Levantar la API
uvicorn app.main:app --reload --port 8000
```

La documentación interactiva de los endpoints queda en <http://localhost:8000/docs>.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Abrir <http://localhost:5173>. Si el backend corre en otro puerto, configurarlo en `frontend/.env`:

```
VITE_API_URL=http://localhost:8000
```

## Problemas comunes

**Error de permisos con .venv en PowerShell**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**El endpoint /metricas devuelve 503**  
Falta ejecutar el paso de evaluación (punto 3 del backend).

**El frontend muestra "Backend Offline"**  
Verificar que el backend esté corriendo en `http://localhost:8000`.

**El puerto 8000 ya está ocupado**
```bash
uvicorn app.main:app --reload --port 8001
# y actualizar VITE_API_URL en frontend/.env
```

## 5. Dataset

**Tipo:** sintético, generado con `backend/data/generate_dataset.py` (semilla fija para reproducibilidad).

Cada fila es un **par de materias**:

| Campo | Tipo | Descripción |
|---|---|---|
| `materia_id_1`, `materia_id_2` | id | par de materias |
| `cant_estudiantes_compartidos_historico` | numérico | estudiantes que cursaron/rindieron ambas históricamente |
| `mismo_anio` | booleano | ambas pertenecen al mismo año de la carrera |
| `misma_carrera` | booleano | ambas pertenecen a la misma carrera |
| `cant_estudiantes_afectados` | numérico | **variable a predecir** (regresión) |
| `hubo_conflicto` | booleano | **variable a predecir** (clasificación) |

### Reglas de generación

- **Estructura académica:** 40 materias en 5 años a lo largo de 3 carreras de ingeniería (con materias de tronco común en 1.º y 2.º año). Total: $\binom{40}{2} = 780$ pares de materias.
- **Tamaño de materias ($S_i$):** Matrícula piramidal decreciente por año (200–350 en 1.º año hasta 20–50 en 5.º año).
- **Estudiantes compartidos históricos:** Acotado por $\min(S_1, S_2)$, modulado por afinidad de carrera ($F_{\text{carrera}}$), cercanía temporal ($F_{\text{año}} = \frac{1}{1 + 0.8 \Delta_{\text{año}}}$) y ruido estocástico.
- **Variable continua a predecir (`cant_estudiantes_afectados`):**
  $$\text{afectados} = \max\!\left(0,\; \min\!\left(\text{compartidos},\; 0.55 \cdot \text{compartidos} + 6.0 \cdot \text{mismo\_anio} + 4.0 \cdot \text{misma\_carrera} + \mathcal{N}(0,\, 2.5^2)\right)\right)$$
- **Variable binaria derivada (`hubo_conflicto`):**
  $$\text{hubo\_conflicto} = (\text{cant\_estudiantes\_afectados} \ge 15)$$
- Semilla fija (`--seed 42`) para reproducibilidad garantizada.

> ⚠️ Como el dataset es sintético, el modelo **va a redescubrir las reglas que pusimos nosotros**. Esto es esperable; las métricas reflejan qué tan bien recupera esa estructura, no el rendimiento sobre datos reales.

## 6. Modelo predictivo

- **Tarea elegida:** regresión sobre `cant_estudiantes_afectados`. Usamos regresión (y no clasificación) porque la salida continua permite que el usuario ajuste el umbral τ desde el frontend para armar el cronograma según la capacidad de la facultad; con una clasificación binaria fija ese ajuste no sería posible.
- **Modelos comparados:** Regresión Lineal (OLS), Árbol de Decisión y K-Nearest Neighbors (KNN).
- **Partición:** 70 % train / 15 % val / 15 % test (semilla 42; el test no se toca hasta la evaluación final).
- **Selección de hiperparámetros:** `GridSearchCV` con K-Fold sobre el conjunto de train.
- **Preprocesamiento:** escalado de variables numéricas con `StandardScaler`, booleanas sin transformación adicional.

### Métricas de comparación sobre validación

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión Lineal (OLS) | **1.68** | **2.14** | **0.9915** |
| Árbol de Decisión (`max_depth=5`) | 2.45 | 3.20 | 0.965 |
| KNN Regressor (`k=7`) | 3.10 | 4.15 | 0.942 |

**Modelo elegido:** Regresión Lineal (OLS) — menor error en validación, inferencia instantánea y los coeficientes son interpretables para la cátedra.

### Métricas sobre el conjunto de test (117 pares, semilla 42)

**Regresión** (`cant_estudiantes_afectados`):

| MAE | RMSE | R² |
|---|---|---|
| 1.68 estudiantes | 2.14 estudiantes | 0.9915 |

**Clasificación derivada** (umbral τ = 15 sobre la predicción continua):

| Accuracy | Precisión | Recall | F1 |
|---|---|---|---|
| 94.87 % | 95.12 % | 90.70 % | 92.86 % |

Matriz de confusión: VN = 72 · FP = 2 · FN = 4 · VP = 39.

**Análisis de errores:** el modelo tiende a subestimar levemente en pares de materias del mismo año con alta matrícula, donde el ruido estocástico del dataset es mayor. Los falsos negativos (FN = 4) corresponden a pares con conflicto real justo en el límite del umbral (afectados ≈ 15–17), que el modelo predice ligeramente por debajo. No hay evidencia de sesgo sistemático por carrera.

## 7. Algoritmo de grafos

### Coloreo greedy (Welsh-Powell)

Se asigna a cada materia el menor turno (color) que no esté usado por ninguna de sus vecinas en el grafo de conflictos.

```
ordenar nodos por grado decreciente (Welsh-Powell)
para cada nodo v en ese orden:
    colores_vecinos = { color[u] : u vecino de v, u ya coloreado }
    color[v] = menor entero ≥ 0 que no esté en colores_vecinos
```

- **Orden de los nodos:** grado decreciente (cantidad de conflictos). En caso de empate, se desempata por suma de pesos de aristas incidentes (estudiantes afectados totales), y si sigue empatado, por ID de materia para garantizar determinismo.
- **Complejidad:** O(V + E) para el coloreo, más O(V log V) del ordenamiento.
- **Garantía:** usa a lo sumo Δ + 1 colores (Δ = grado máximo). No garantiza el mínimo (el coloreo óptimo es NP-difícil), pero en la práctica da buenos resultados.
- **Validación:** se verifica que ninguna arista tenga extremos del mismo color.

El algoritmo está implementado a mano en `backend/app/graph/coloring.py`; `networkx` se usa solo como estructura de datos para representar el grafo.

## 8. Integración modelo → grafo

1. El modelo predice, para cada par de materias, el conflicto esperado (cantidad continua de estudiantes afectados).
2. Se crea una arista entre dos materias si el conflicto predicho **supera el umbral** `τ` (configurable desde el frontend, entre 0 y 100).
3. El **peso** de la arista es el valor predicho (estudiantes afectados), usado para visualización y para desempatar el orden de coloreo.
4. Se colorea el grafo resultante. La cantidad de colores es la cantidad de turnos asignados.

**Por qué es razonable:** un umbral bajo agrega aristas (más restricciones → más turnos, menos estudiantes afectados); un umbral alto quita aristas (menos turnos, pero quedan conflictos menores sin resolver). El umbral expresa el trade-off entre **cantidad de turnos** y **estudiantes perjudicados**.

## 9. API del backend

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servidor |
| `GET` | `/materias` | Lista de materias (nodos del grafo) |
| `GET` | `/conflictos?umbral=τ` | Pares con conflicto predicho ≥ τ |
| `POST` | `/cronograma` | Construye el grafo y devuelve el coloreo |
| `GET` | `/metricas` | Métricas del modelo desde `metricas.json` |

`GET /metricas` devuelve los metadatos del modelo, métricas de regresión y clasificación, y la matriz de confusión desde `backend/models/metricas.json`. Responde `503` si el archivo todavía no está disponible y `500` si su contenido es inválido.

Ejemplo — `POST /cronograma`:

```bash
curl -X POST http://localhost:8000/cronograma \
  -H "Content-Type: application/json" \
  -d '{"umbral": 15}'
```

```json
{
  "umbral": 15,
  "cantidad_turnos": 6,
  "cantidad_nodos": 40,
  "cantidad_aristas": 48,
  "nodos": [
    { "id": "M01", "nombre": "Análisis Matemático I", "turno": 0, "anio": 1, "carreras": ["ISI", "IQ"] }
  ],
  "aristas": [
    { "source": "M01", "target": "M07", "peso": 23.4 }
  ],
  "turnos": {
    "0": ["M01", "M05", "M12"],
    "1": ["M02", "M08"]
  },
  "es_valido": true
}
```

## 10. Frontend

React 19 + TypeScript + Vite. Consume la API del backend.

Las llamadas HTTP se centralizan en `frontend/src/api.ts`, con tipos compartidos en `frontend/src/types.ts` y manejo reutilizable de estados de carga y error en `frontend/src/useApiRequest.ts`.

El grafo de conflictos se visualiza con Cytoscape.js: cada materia se colorea por turno y el grosor de las aristas representa el peso del conflicto. La visualización valida que los extremos de cada arista tengan turnos distintos, muestra el peso exacto al pasar el cursor o seleccionar una arista y permite reajustar el encuadre. El componente acepta un umbral configurable y actualiza los elementos del grafo sin reconstruir el layout.

El selector de umbral (`frontend/src/UmbralSelector.tsx`) permite variar $\tau$ interactivamente con debounce (350 ms) y botones de acceso rápido (presets: 5, 15, 30, 50), actualizando el grafo, el coloreo y la cantidad de turnos sin congelar la interfaz.

La vista de cronograma (`frontend/src/CronogramaGrid.tsx`) organiza las materias en columnas por turno, sincronizada exactamente con los colores determinísticos del grafo (`frontend/src/colors.ts`). Muestra la cantidad de materias por turno, métricas globales de distribución y un buscador rápido para localizar materias específicas.

El panel de métricas (`frontend/src/MetricsPanel.tsx`) resume en tiempo real los KPIs del cronograma (cantidad de turnos, aristas activas, impacto estimado de conflictos, densidad del grafo) junto con las métricas de evaluación del modelo de ML (MAE, RMSE, $R^2$, Accuracy, matriz de confusión y metadatos de entrenamiento).

- **Grafo de conflictos:** materias como nodos coloreados según el turno asignado; grosor de arista según el peso.
- **Vista de cronograma:** grilla/calendario con las materias de cada turno.
- **Selector de umbral:** slider que recalcula grafo, coloreo y cantidad de turnos.
- **Panel de métricas:** KPIs dinámicos del cronograma y métricas de evaluación del modelo de ML.

_Capturas de pantalla en `informe/capturas/`._

## 11. Tests

```bash
cd backend
pytest
```

La suite cubre 166 casos en total. Se prueban, entre otras cosas:

- el coloreo nunca asigna el mismo turno a dos materias adyacentes (property-based sobre grafos aleatorios);
- subir el umbral nunca aumenta la cantidad de aristas (invariante de monotonía);
- el grafo vacío usa un único turno;
- el endpoint `/cronograma` devuelve una estructura válida con los campos correctos.

## 12. Uso de IA

Usamos asistentes de IA (principalmente Claude y GitHub Copilot) a lo largo del proyecto:

- **Herramientas:** Claude (Anthropic) y GitHub Copilot.
- **Usada para:** generar el esqueleto inicial del frontend en React, sugerir la estructura de los tests de pytest, debuggear errores puntuales de Cytoscape.js (props que cambiaron entre versiones), y reformular las descripciones del README.
- **No usada para:** el diseño del algoritmo de coloreo greedy, la lógica de predicción del modelo, la arquitectura del sistema ni las decisiones de diseño de los endpoints.
- **Casos en que se equivocó:** propuso nombres de props de Cytoscape que no existían en la versión que instalamos, generó imports incorrectos en TypeScript que tuvimos que corregir a mano, y en un momento sugirió presentar el coloreo greedy como "óptimo" cuando no lo es (lo corregimos).

## 13. Limitaciones y extensiones

**Limitaciones conocidas**

- Dataset sintético: las métricas no se extrapolan a datos reales.
- El coloreo greedy no garantiza el número cromático (cantidad mínima de turnos posible).
- No se modelan restricciones de aulas, docentes ni capacidad por turno.

**Posibles extensiones**

- Coloreo DSATUR o con backtracking para acercarse al óptimo.
- Restricciones de capacidad de aulas y de docentes.
- Entrenar con datos reales de inscripciones.
- Comparar varios umbrales automáticamente y sugerir el mejor.

## 14. Fuentes y licencia

**Librerías utilizadas:**

| Área | Librerías |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic v2, NetworkX, Scikit-Learn, Pandas, NumPy, Joblib |
| Frontend | React 19, TypeScript, Vite, Cytoscape.js |
| Tests | Pytest, HTTPX |
| Visualización | Matplotlib, Seaborn |

**Origen de los datos:** dataset sintético generado mediante simulación estocástica con semilla fija (`--seed 42`), basado en la estructura de planes de estudio reales de carreras de ingeniería de UTN. No se usaron datos reales de estudiantes.

**Licencia:** MIT License — proyecto académico sin fines de lucro, UTN FRRo, 2026.
