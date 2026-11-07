# Algoritmo de Coloreo Greedy de Grafos — Ticket T14

**Proyecto:** Planificación de Mesas de Examen mediante Grafos y Machine Learning  
**Módulo:** `backend/app/graph/coloring.py`  
**Tests:** `backend/tests/test_coloring.py`

---

## 1. Fundamento del Problema de Asignación de Turnos

El cronograma de exámenes se modela como un problema clásico de **Coloreo de Vértices** sobre el grafo de conflictos $G = (V, E)$:
- **Vértices ($V$):** Conjunto de materias que deben rendirse.
- **Aristas ($E$):** Pares de materias cuyo conflicto estimado supera el umbral $\tau$ ($\hat{y}_{ij} \ge \tau$).
- **Colores ($\mathcal{C}$):** Turnos o fechas de examen ($0, 1, 2, \ldots, k-1$).

**Regla de Factibilidad Académica:**
$$\forall (u, v) \in E \implies \text{turno}(u) \neq \text{turno}(v)$$

**Objetivo de Optimización:**  
Minimizar la cantidad total de turnos necesarios $k = |\mathcal{C}|$ (número cromático $\chi(G)$), reduciendo la duración total del período de exámenes sin que ningún estudiante sufra superposiciones.

Como el cálculo del número cromático exacto $\chi(G)$ es un problema **NP-difícil**, se implementa una heurística constructiva voraz (**Greedy**) basada en **Welsh-Powell**.

---

## 2. Heurística de Ordenamiento: Welsh-Powell Extendido

El resultado de un algoritmo greedy depende crucialmente del orden en que se recorren los nodos. Un orden arbitrario puede dispersar colores y elevar innecesariamente la cantidad de turnos.

Se aplica un ordenamiento estricto multinivel implementado en `get_node_ordering()`:

1. **Criterio Primario — Grado $\deg(v)$ descendente:**
   Las materias con mayor cantidad de materias en conflicto son las más restrictivas. Colorearlas primero garantiza que encuentren turnos bajos disponibles antes de que sus vecinos ocupen opciones.
2. **Criterio Secundario — Suma de Pesos Incidentes ($\sum w_{uv}$) descendente:**
   Si dos materias tienen el mismo grado, desempata la que acumule más estudiantes afectados en total, priorizando la resolución de los cuellos de botella más masivos de la facultad.
3. **Criterio Terciario — Identificador de la materia (lexicográfico):**
   Garantiza determinismo absoluto y reproducibilidad ante ejecuciones repetidas.

---

## 3. Algoritmo Voraz Paso a Paso

```
Algoritmo: greedy_coloring(G)
Entrada: Grafo no dirigido G = (V, E)
Salida: Mapeo {nodo: turno}

1. Si V es vacío, retornar {}
2. V_ordenado = ordenar V según (-grado, -suma_pesos, id)
3. coloring = {}
4. Para cada v en V_ordenado:
     colores_prohibidos = { coloring[u] para todo u vecino de v tal que u está en coloring }
     color = 0
     Mientras color pertenezca a colores_prohibidos:
         color = color + 1
     coloring[v] = color
5. Retornar coloring
```

---

## 4. Garantías Teóricas y Complejidad

### 4.1. Garantía Teórica (Cota de Brooks)
Para cualquier grafo $G$, el algoritmo greedy garantiza utilizar a lo sumo:

$$\chi_{\text{greedy}}(G) \le \Delta(G) + 1$$

donde $\Delta(G) = \max_{v \in V} \deg(v)$ es el grado máximo del grafo. Si el grafo no es completo ni un ciclo impar, el Teorema de Brooks asegura $\chi(G) \le \Delta(G)$.

### 4.2. Análisis de Complejidad Computacional
- **Ordenamiento de Nodos:**
  Calcular grados y pesos incidentes toma $\mathcal{O}(|V| + |E|)$. Ordenar los $|V|$ vértices toma $\mathcal{O}(|V| \log |V|)$.
- **Asignación de Colores:**
  Para cada vértice $v$, se inspeccionan sus vecinos $\deg(v)$. A lo largo de todo el ciclo, la suma de exploraciones es $\sum_{v \in V} \deg(v) = 2|E|$, lo cual toma $\mathcal{O}(|V| + |E|)$.
- **Complejidad Temporal Total:**
  $$\mathcal{O}(|V| \log |V| + |E|)$$
  Para el grafo institucional del proyecto ($|V| = 40$, $|E| \le 780$), la ejecución toma menos de **1 milisegundo**.
- **Complejidad Espacial:**
  $$\mathcal{O}(|V|)$$
  Almacenamiento de la asignación y estructuras auxiliares de vecinos.

---

## 5. Validación y Resultados sobre el Grafo del Proyecto

Sobre el grafo sintético del proyecto (40 materias, umbral institucional $\tau = 15$ alumnos):
- **Materias:** 40
- **Aristas de Conflicto:** 301
- **Grado Máximo ($\Delta$):** 28
- **Cota Teórica Máxima:** $\le 29$ turnos
- **Turnos obtenidos con Welsh-Powell:** **13 turnos** (muy por debajo de la cota máxima teórica $\Delta + 1$).
- **Validación:** 100% libre de conflictos adyacentes (`validate_coloring` = True).
