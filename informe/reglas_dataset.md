# Especificación de Reglas de Generación del Dataset Sintético (T04)

Este documento detalla el diseño formal y las reglas matemáticas que gobiernan la generación del dataset sintético de pares de materias (`pares_materias.csv`), a ser implementado en el script `backend/data/generate_dataset.py`.

---

## 1. Definición del Universo Académico

El modelo simula una facultad universitaria con carreras de ingeniería y un conjunto representativo de materias distribuidas por año académico.

### 1.1. Carreras y Años
* **Cantidad de Carreras ($C$):** 3 carreras:
  * `C1`: Ingeniería en Sistemas
  * `C2`: Ingeniería Industrial
  * `C3`: Ingeniería Química
* **Años Lectivos ($A$):** 5 años (del 1.º al 5.º año).

### 1.2. Cantidad de Materias ($N$)
* **Por defecto:** $N = 40$ materias (configurable mediante CLI vía `--n-materias`).
* **Distribución de Materias:**
  * **1.º y 2.º Año (Ciencias Básicas / Tronco Común):** ~16 materias (8 por año). Son materias compartidas o transversales (Álgebra, Análisis Matemático, Física, Química General, etc.) a las que concurren estudiantes de múltiples carreras.
  * **3.º, 4.º y 5.º Año (Especialidad):** ~24 materias distribuidas entre las 3 carreras (~8 materias por carrera, especializadas por año).
* **Cantidad de Pares Únicos Generados:**
  Para $N = 40$, el dataset contiene todos los pares no dirigidos sin repetición ni auto-pares:
  $$\binom{N}{2} = \frac{40 \times 39}{2} = 780 \text{ pares}$$

---

## 2. Tamaño de las Materias (Inscriptos Activos)

En el ámbito universitario, la matrícula disminuye a medida que se avanza en la carrera (estructura piramidal de cursada).

El tamaño $S_i$ de una materia $i$ se genera a partir de una base esperada según su año lectivo $\text{anio}_i$ más una perturbación gaussiana:

$$S_i = \max\left(20, \, \text{round}\left(\text{Base}(\text{anio}_i) + \epsilon_{\text{tam}}\right)\right), \quad \epsilon_{\text{tam}} \sim \mathcal{N}(0, \sigma^2)$$

| Año ($\text{anio}_i$) | Matrícula Base esperada | Rango típico de alumnos |
|---|---|---|
| **1.º Año** | 250 | 200 – 350 |
| **2.º Año** | 160 | 120 – 220 |
| **3.º Año** | 90 | 60 – 140 |
| **4.º Año** | 50 | 35 – 80 |
| **5.º Año** | 30 | 20 – 50 |

---

## 3. Características del Par (Features)

Para cada par ordenado de materias $(M_1, M_2)$ con $id(M_1) < id(M_2)$:

1. **`mismo_anio` (Booleano):**
   $$\text{mismo\_anio} = (\text{anio}_{M_1} == \text{anio}_{M_2})$$

2. **`misma_carrera` (Booleano):**
   $$\text{misma\_carrera} = (\text{carrera}_{M_1} \cap \text{carrera}_{M_2} \neq \emptyset)$$
   *(Verdadero si comparten al menos una carrera o ambas son materias básicas comunes).*

---

## 4. Modelado de Estudiantes Compartidos Históricos

La variable `cant_estudiantes_compartidos_historico` mide la cantidad acumulada de estudiantes que han cursado o regularizado ambas asignaturas simultáneamente o de forma secuencial.

### 4.1. Techo Físico
Dos materias nunca pueden compartir más alumnos que la totalidad de inscriptos de la materia con menor matrícula:
$$\text{Techo} = \min(S_{M_1}, S_{M_2})$$

### 4.2. Factores de Afinidad
La proporción de estudiantes compartidos depende de dos dimensiones clave:
1. **Factor de Carrera ($F_{\text{carrera}}$):**
   * Si $\text{misma\_carrera} = \text{True}$: $F_{\text{carrera}} \sim \mathcal{U}(0.60, 0.85)$
   * Si $\text{misma\_carrera} = \text{False}$: $F_{\text{carrera}} \sim \mathcal{U}(0.01, 0.05)$ (cruces marginales o cambios de carrera).

2. **Factor de Proximidad Temporal ($F_{\text{año}}$):**
   Los estudiantes comparten más materias si pertenecen al mismo año o a años consecutivos:
   $$\Delta_{\text{año}} = |\text{anio}_{M_1} - \text{anio}_{M_2}|$$
   $$F_{\text{año}} = \frac{1}{1 + 0.8 \times \Delta_{\text{año}}}$$

### 4.3. Ecuación de Compartidos
$$\mu_{\text{compartidos}} = \min(S_{M_1}, S_{M_2}) \times F_{\text{carrera}} \times F_{\text{año}}$$
$$\text{cant\_estudiantes\_compartidos\_historico} = \max\left(0, \, \min\left(\text{Techo}, \, \text{round}\left(\mu_{\text{compartidos}} + \epsilon_{\text{comp}}\right)\right)\right)$$
donde $\epsilon_{\text{comp}} \sim \mathcal{N}(0, 3^2)$ introduce ruido estocástico no determinista.

---

## 5. Variables Objetivo (Targets)

### 5.1. Regresión: `cant_estudiantes_afectados`
Estima la cantidad real de estudiantes que tendrían un conflicto directo si las mesas de examen de $M_1$ y $M_2$ se fijan en el mismo turno y día.

La relación lineal con las variables y factores del dominio se formula como:

$$\text{afectados} = \alpha \cdot \text{cant\_estudiantes\_compartidos\_historico} + \beta \cdot \mathbb{I}_{\text{mismo\_anio}} + \gamma \cdot \mathbb{I}_{\text{misma\_carrera}} + \epsilon_{\text{afect}}$$

* **Ponderaciones:**
  * $\alpha = 0.55$: aproximadamente el 55% de los estudiantes que comparten materias rinden en la misma época.
  * $\beta = 6.0$: incremento por regularidad simultánea en el mismo año lectivo.
  * $\gamma = 4.0$: incremento por pertenecer al mismo plan de estudios de la carrera.
  * $\epsilon_{\text{afect}} \sim \mathcal{N}(0, 2.5^2)$: ruido de medición / comportamiento aleatorio de inscripción.

* **Restricciones de consistencia:**
  $$0 \le \text{cant\_estudiantes\_afectados} \le \text{cant\_estudiantes\_compartidos\_historico}$$

### 5.2. Clasificación: `hubo_conflicto`
Variable binaria indicativa de si la cantidad de estudiantes perjudicados supera el umbral de tolerancia académica institucional $k$:

$$\text{hubo\_conflicto} = (\text{cant\_estudiantes\_afectados} \ge k)$$

* **Umbral de corte:** $k = 15$ estudiantes.
* Este umbral produce una partición aproximada de 30% a 35% de pares en conflicto (`True`), logrando un balance adecuado para entrenamiento y métricas (F1-score, precisión, recall, matriz de confusión).

---

## 6. Estructura del Dataset Resultante

| Campo | Tipo | Ejemplo | Descripción |
|---|---|---|---|
| `materia_id_1` | `str` | `"M01"` | Identificador único de la primera materia |
| `materia_id_2` | `str` | `"M08"` | Identificador único de la segunda materia |
| `cant_estudiantes_compartidos_historico` | `int` | `48` | Estudiantes con ambas materias en historial |
| `mismo_anio` | `bool` | `True` | Pertenecen al mismo año de cursada |
| `misma_carrera` | `bool` | `True` | Pertenecen a la misma carrera o tronco común |
| `cant_estudiantes_afectados` | `float` / `int` | `32` | **Target Regresión**: estudiantes afectados por superposición |
| `hubo_conflicto` | `bool` | `True` | **Target Clasificación**: si `cant_estudiantes_afectados >= 15` |

---

## 7. Advertencia Metodológica (Para el Informe)

> [!WARNING]
> Como el dataset es sintético, los modelos de Machine Learning (Regresión Lineal, Árboles de Decisión, KNN) **redescubrirán las reglas que definimos en este documento**.
> 
> Por lo tanto:
> 1. Un $R^2$ alto o un $F1$ elevado refleja la capacidad del modelo de reconstruir la función generadora frente al ruido introducido ($\epsilon_{\text{comp}}$, $\epsilon_{\text{afect}}$).
> 2. Las métricas obtenidas no deben interpretarse como rendimiento sobre datos reales de alumnos universitarios sin un proceso de validación externa.

