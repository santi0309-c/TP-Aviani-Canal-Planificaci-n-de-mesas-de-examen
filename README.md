# TP-Aviani-Canal-Planificacion-de-mesas-de-examen

**Integrantes:** Valentino Aviani · Santiago Cañal

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
3. **Asigna** turnos de examen mediante **coloreo greedy**, usando la menor cantidad de turnos posible.
4. **Muestra** el resultado en una aplicación web donde se puede variar el umbral de conflicto y ver cómo cambia el cronograma.
## 2. Cómo funciona (visión general)
 
```
dataset de pares ──► modelo (regresión/clasificación) ──► conflicto predicho por par
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
Si el modelo cambia su predicción (o se cambia el umbral), el grafo cambia y el coloreo se recalcula.
 
## 3. Estructura del repositorio
 
> Ajustar a la estructura real del proyecto.
 
```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # API (FastAPI)
│   │   ├── schemas.py           # modelos Pydantic de entrada/salida
│   │   ├── ml/
│   │   │   ├── features.py      # preparación de datos / features
│   │   │   ├── train.py         # entrenamiento y comparación de modelos
│   │   │   └── predict.py       # carga del modelo y predicción
│   │   └── graph/
│   │       ├── build.py         # construcción del grafo de conflictos
│   │       └── coloring.py      # coloreo greedy
│   ├── data/
│   │   ├── generate_dataset.py  # script generador del dataset sintético
│   │   └── pares_materias.csv   # dataset generado
│   ├── models/                  # modelo entrenado (.joblib)
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   └── package.json
├── informe/                     # informe y capturas
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
 
# 3) Levantar la API
uvicorn app.main:app --reload --port 8000
```
 
La documentación interactiva de la API queda en <http://localhost:8000/docs>.
 
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
 
### Atajo (opcional)
 
```bash
make setup   # instala dependencias de backend y frontend
make data    # genera dataset y entrena el modelo
make run     # levanta backend y frontend
```
 
## 5. Dataset
 
**Tipo:** sintético, generado con `backend/data/generate_dataset.py`.
_(Si se usó un dataset real, indicar fuente, link, fecha de descarga y transformaciones aplicadas.)_
 
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
 
> Completar con las reglas reales del script. Ejemplo de cómo declararlas:
 
- La cantidad de estudiantes compartidos histórica depende del tamaño de las materias, de si son del mismo año (+) y de si son de la misma carrera (++), más ruido.
- `cant_estudiantes_afectados` ≈ `α · compartidos + β · mismo_anio + γ · misma_carrera + ruido`.
- `hubo_conflicto = cant_estudiantes_afectados ≥ k`.
- Semilla fija (`--seed`) para reproducibilidad.
> ⚠️ Como el dataset es sintético, el modelo **va a redescubrir las reglas que pusimos nosotros**. Esto es esperable; las métricas reflejan qué tan bien recupera esa estructura, no el rendimiento sobre datos reales.
 
## 6. Modelo predictivo
 
- **Tarea:** _regresión (`cant_estudiantes_afectados`) / clasificación (`hubo_conflicto`)_ — elegir una y justificar.
- **Modelos comparados:** _regresión lineal, árbol de decisión, KNN, …_
- **Partición:** _70 % train / 15 % val / 15 % test_ (semilla fija; el test no se toca hasta la evaluación final).
- **Selección de hiperparámetros:** _validación cruzada k-fold sobre train_.
- **Preprocesamiento:** _escalado de variables numéricas, codificación de booleanas_.
### Métricas (conjunto de test)
 
> Completar con los resultados reales.
 
| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Regresión lineal | | | |
| Árbol de decisión | | | |
| KNN | | | |
 
_(Si es clasificación: accuracy, precisión, recall, F1 y matriz de confusión.)_
 
**Modelo elegido:** _…_ — **motivo:** _…_
 
**Análisis de errores:** _en qué tipo de pares se equivoca más, si sub/sobreestima, etc._
 
## 7. Algoritmo de grafos
 
### Coloreo greedy
 
Se asigna a cada materia el menor turno (color) que no esté usado por ninguna de sus vecinas en el grafo de conflictos.
 
```
ordenar los nodos según una heurística
para cada nodo v en ese orden:
    colores_vecinos = { color[u] : u vecino de v, u ya coloreado }
    color[v] = menor entero ≥ 0 que no esté en colores_vecinos
```
 
- **Orden de los nodos:** _por grado decreciente (Welsh-Powell) / DSATUR / otro_.
- **Complejidad:** O(V + E) para el coloreo, más O(V log V) del ordenamiento.
- **Garantía:** usa a lo sumo Δ + 1 colores (Δ = grado máximo). No garantiza el mínimo (el coloreo óptimo es NP-difícil), pero en la práctica da buenos resultados.
- **Validación:** se verifica que ninguna arista tenga extremos del mismo color.
El algoritmo está implementado a mano en `backend/app/graph/coloring.py`; `networkx` se usa solo para representar el grafo. _(Ajustar si se usó otra cosa.)_
 
## 8. Integración modelo → grafo
 
1. El modelo predice, para cada par de materias, el conflicto esperado.
2. Se crea una arista entre dos materias si el conflicto predicho **supera el umbral** `τ` (configurable desde el frontend).
3. El **peso** de la arista es el valor predicho (estudiantes afectados), usado para visualización y para desempatar el orden de coloreo.
4. Se colorea el grafo resultante. La cantidad de colores es la cantidad de turnos necesarios.
**Por qué es razonable:** un umbral bajo agrega aristas (más restricciones → más turnos, menos estudiantes afectados); un umbral alto quita aristas (menos turnos, pero quedan conflictos menores sin resolver). El umbral expresa el trade-off entre **cantidad de turnos** y **estudiantes perjudicados**.
 
## 9. API del backend
 
> Ajustar rutas y campos a la implementación real.
 
| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/materias` | Lista de materias (nodos) |
| `GET` | `/conflictos?umbral=τ` | Pares con conflicto predicho ≥ τ |
| `POST` | `/cronograma` | Construye el grafo y devuelve el coloreo |
| `GET` | `/metricas` | Métricas del modelo sobre test |
 
Ejemplo:
 
```bash
curl -X POST http://localhost:8000/cronograma \
  -H "Content-Type: application/json" \
  -d '{"umbral": 15}'
```
 
```json
{
  "umbral": 15,
  "cantidad_turnos": 5,
  "nodos": [{ "id": "M01", "nombre": "Análisis I", "turno": 0 }],
  "aristas": [{ "source": "M01", "target": "M07", "peso": 23.4 }]
}
```
 
## 10. Frontend
 
React + TypeScript. Consume la API del backend.
 
- **Grafo de conflictos:** materias como nodos coloreados según el turno asignado; grosor de arista según el peso.
- **Vista de cronograma:** grilla/calendario con las materias de cada turno.
- **Selector de umbral:** slider que recalcula grafo, coloreo y cantidad de turnos.
_Capturas de pantalla en `informe/capturas/`._
 
## 11. Tests
 
```bash
cd backend
pytest
```
 
Se prueban, como mínimo:
 
- el coloreo nunca asigna el mismo turno a dos materias adyacentes;
- subir el umbral nunca aumenta la cantidad de aristas;
- el grafo vacío usa un único turno;
- el endpoint `/cronograma` devuelve una estructura válida.
## 12. Uso de IA
 
Se detalla en el informe (sección "Uso de IA"). Resumen:
 
- **Herramientas:** _…_
- **Usada para:** _script generador del dataset, esqueleto del frontend, …_
- **No usada para:** _…_
- **Casos en que se equivocó:** _…_
## 13. Limitaciones y extensiones
 
**Limitaciones conocidas**
 
- Dataset sintético: las métricas no se extrapolan a datos reales.
- El coloreo greedy no garantiza el número cromático.
- No se modelan restricciones de aulas, docentes ni capacidad por turno.
**Posibles extensiones**
 
- Coloreo DSATUR o con backtracking para acercarse al óptimo.
- Restricciones de capacidad de aulas y de docentes.
- Entrenar con datos reales de inscripciones.
- Comparar varios umbrales automáticamente y sugerir el mejor.
## 14. Fuentes y licencia
 - Librerías: scikit-learn, pandas, networkx, FastAPI, React.
- _Si se usó un dataset real, citarlo acá._
- Proyecto académico. _(Agregar licencia si corresponde.)_
- Librerías: scikit-learn, pandas, networkx, FastAPI, React.
- _Si se usó un dataset real, citarlo acá._
- Proyecto académico. _(Agregar licencia si corresponde.)_
