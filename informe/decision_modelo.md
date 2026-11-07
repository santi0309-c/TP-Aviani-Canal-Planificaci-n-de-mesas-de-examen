# Decisión de Modelado: Regresión vs. Clasificación — Ticket T07

**Proyecto:** Planificación de Mesas de Examen mediante Grafos y Machine Learning  
**Autores:** Valentino Aviani · Santiago Cañal  
**Dataset de referencia:** `backend/data/pares_materias.csv` (780 pares, $\binom{40}{2}$)  
**Fecha:** Noviembre 2026

---

## 1. Planteo del Dilema

El pipeline del sistema requiere que un modelo de Machine Learning procese cada par de materias $(M_i, M_j)$ y produzca una salida que alimente la construcción del **grafo de conflictos**. El dataset generado en T05 ofrece dos variables objetivo candidatas:

| Variable | Tipo | Rango | Descripción |
|---|---|---|---|
| `cant_estudiantes_afectados` | `int` (continuo) | $[0, 123]$ | Estudiantes perjudicados si ambas mesas coinciden |
| `hubo_conflicto` | `bool` (binario) | $\{0, 1\}$ | Si `cant_estudiantes_afectados` $\ge 15$ |

La elección condiciona directamente:
- La **función de pérdida** y las **métricas de evaluación** (T09 / T10).
- La **construcción del grafo** (T13): aristas con peso continuo vs. aristas binarias sin peso.
- La **experiencia de usuario** (T22): si el frontend puede ofrecer un umbral de tolerancia $\tau$ dinámico o si el corte queda fijo.

---

## 2. Análisis Comparativo

### 2.1. Opción A — Regresión sobre `cant_estudiantes_afectados`

El modelo predice directamente $\hat{y}_{ij} \in \mathbb{R}_{\ge 0}$: la cantidad estimada de estudiantes afectados para el par $(M_i, M_j)$.

| | Detalle |
|---|---|
| **Pro 1 — Granularidad informativa** | Se conserva la magnitud del conflicto. Un par con $\hat{y} = 90$ (Álgebra vs. Física I) es cualitativamente distinto de uno con $\hat{y} = 16$ (apenas por encima del umbral). La regresión preserva esa diferencia; la clasificación la destruye al colapsar ambos a `True`. |
| **Pro 2 — Umbral dinámico $\tau$** | La decisión binaria *"¿hay conflicto?"* se posterga al momento de construir el grafo, aplicando $A_{ij} = \mathbb{I}[\hat{y}_{ij} \ge \tau]$ con un $\tau$ configurable. Esto permite al usuario del frontend ajustar la sensibilidad del cronograma sin reentrenar el modelo. |
| **Pro 3 — Pesos en aristas** | El valor continuo $\hat{y}_{ij}$ sirve directamente como peso $w_{ij}$ de la arista en el grafo. Algoritmos de coloreo pesado (Weighted DSatur) pueden priorizar la separación de los pares más críticos, produciendo cronogramas de mayor calidad. |
| **Pro 4 — Compatibilidad con métricas de negocio** | Si la institución cambia su política de tolerancia (de 15 a 20 alumnos, por ejemplo), basta con mover $\tau$ sin reentrenar ni recolectar datos. |
| **Contra 1 — Sensibilidad a outliers** | Las distribuciones de `cant_estudiantes_compartidos_historico` y `cant_estudiantes_afectados` presentan asimetría positiva pronunciada (mediana = 10, máximo = 123). Valores extremos de pares de materias básicas masivas pueden inflar el MAE y el RMSE, dificultando la interpretación directa de las métricas de error. |
| **Contra 2 — Métricas menos intuitivas para la gestión** | Un $R^2 = 0.95$ o un $\text{MAE} = 3.2$ no comunican directamente al tomador de decisiones académico si el cronograma va a funcionar o no; requieren una traducción posterior a impacto institucional. |

### 2.2. Opción B — Clasificación sobre `hubo_conflicto`

El modelo predice directamente $\hat{c}_{ij} \in \{0, 1\}$: si el par tiene o no conflicto institucional.

| | Detalle |
|---|---|
| **Pro 1 — Decisión directa** | La salida es inmediatamente accionable: `True` → arista en el grafo, `False` → sin arista. No requiere interpretación adicional. |
| **Pro 2 — Métricas orientadas a la decisión** | F1-Score, Precisión, Recall y la matriz de confusión son directamente interpretables por la gestión académica. Un falso negativo (no detectar un conflicto real) tiene un costo operativo claro: estudiantes que pierden una mesa. |
| **Pro 3 — Balance de clases favorable** | El EDA (T06) reveló una proporción de 61.41% (`False`) vs. 38.59% (`True`). Este balance permite entrenar clasificadores estándar sin necesidad de técnicas de sobremuestreo (SMOTE) ni ajuste de pesos de clase. |
| **Contra 1 — Pérdida irreversible de información** | Al binarizar, se destruye la diferencia entre un par con 16 afectados y uno con 120. Ambos son `True`, pero el segundo es **7.5× más crítico**. El grafo resultante no puede distinguirlos ni priorizarlos en el coloreo. |
| **Contra 2 — Umbral fijado en el entrenamiento** | El corte $k = 15$ queda "horneado" en las etiquetas de entrenamiento. Si la institución decide que $k = 10$ o $k = 20$ es más apropiado, hay que **regenerar el dataset, re-etiquetar y reentrenar** todo el pipeline. |
| **Contra 3 — Incompatibilidad con coloreo pesado** | Sin pesos en las aristas, el algoritmo de coloreo greedy trata todos los conflictos como equivalentes, perdiendo la oportunidad de optimizar la separación de los pares más graves. |

### 2.3. Tabla Resumen

| Criterio | Regresión (Opción A) | Clasificación (Opción B) |
|---|:---:|:---:|
| Granularidad de la predicción | ✅ Continua | ❌ Binaria |
| Umbral $\tau$ dinámico en frontend | ✅ Sí, post-predicción | ❌ No, fijado en entrenamiento |
| Aristas ponderadas en el grafo | ✅ $w_{ij} = \hat{y}_{ij}$ | ❌ Sin peso |
| Flexibilidad ante cambios de política | ✅ Solo mover $\tau$ | ❌ Reentrenar |
| Interpretabilidad directa de métricas | ⚠️ Requiere traducción | ✅ F1 / Recall intuitivos |
| Sensibilidad a outliers | ⚠️ Media > Mediana | ✅ Robusto |
| Balance de clases / sesgos | ✅ No aplica | ✅ 61/39 favorable |
| Calidad del cronograma (coloreo pesado) | ✅ Óptima | ⚠️ Limitada |

---

## 3. Decisión: Regresión como Modelo Primario

**Se adopta la Opción A (Regresión)** para predecir `cant_estudiantes_afectados` como variable objetivo principal del pipeline de Machine Learning.

### 3.1. Justificación Técnica

La razón central es que la **regresión preserva información que la clasificación destruye**, y esa información es precisamente la que distingue un cronograma de exámenes bueno de uno óptimo:

1. **El problema de negocio es inherentemente continuo.** La pregunta real de la secretaría académica no es binaria (*"¿hay o no conflicto?"*), sino cuantitativa: *"¿cuántos estudiantes se perjudican si ponemos estas dos materias juntas?"*. La regresión responde directamente a esa pregunta.

2. **La clasificación se obtiene gratis a partir de la regresión.** Dada la predicción continua $\hat{y}_{ij}$, la decisión binaria se deriva trivialmente:
   $$\hat{c}_{ij} = \mathbb{I}\left[\hat{y}_{ij} \ge \tau\right]$$
   Es decir, la regresión **subsume** a la clasificación. Lo inverso no es posible: de una etiqueta binaria no se puede recuperar la magnitud del conflicto.

3. **El umbral dinámico $\tau$ es un requisito funcional del sistema.** El README del proyecto y la arquitectura definida en T13/T22 contemplan explícitamente un selector de umbral en el frontend. Solo la regresión habilita esta funcionalidad sin reentrenamiento.

4. **Los pesos del grafo mejoran la calidad del cronograma.** Con aristas ponderadas por $w_{ij} = \hat{y}_{ij}$, el algoritmo de coloreo puede priorizar la separación de los conflictos más graves (100+ afectados) por encima de los marginales (16-20 afectados), produciendo asignaciones de turnos que minimizan el perjuicio total para los estudiantes.

> [!TIP]
> **Estrategia dual de evaluación:** Aunque el modelo se entrena como regresor, en T10 se evaluará **también** su desempeño como clasificador derivado (aplicando $\tau = 15$ sobre las predicciones), reportando F1-Score, Recall y la matriz de confusión. Esto permite comparar directamente con un clasificador puro y demostrar que no se sacrifica rendimiento de clasificación al elegir regresión.

### 3.2. Traducción Formal al Grafo de Conflictos

Sea $G = (V, E, w)$ el grafo de conflictos donde:
- $V = \{M_1, M_2, \ldots, M_{40}\}$ es el conjunto de materias (nodos).
- $E \subseteq V \times V$ es el conjunto de aristas (pares en conflicto).
- $w: E \to \mathbb{R}_{> 0}$ es la función de peso.

Dado el umbral de tolerancia institucional $\tau \in \mathbb{R}_{> 0}$ (configurable en el frontend, por defecto $\tau = 15$):

$$E = \left\{(M_i, M_j) \;\middle|\; \hat{y}_{ij} \ge \tau,\; i < j \right\}$$

$$w(M_i, M_j) = \hat{y}_{ij} \quad \forall (M_i, M_j) \in E$$

La **matriz de adyacencia ponderada** queda definida como:

$$A_{ij} = \begin{cases} \hat{y}_{ij} & \text{si } \hat{y}_{ij} \ge \tau \\ 0 & \text{en otro caso} \end{cases}$$

> [!NOTE]
> **Propiedad clave:** al variar $\tau$, el grafo se densifica ($\tau \downarrow$, más aristas, más turnos necesarios pero menos riesgo) o se relaja ($\tau \uparrow$, menos aristas, menos turnos pero mayor riesgo). El cronograma se recalcula en tiempo real aplicando el coloreo greedy sobre el nuevo grafo, sin invocar nuevamente al modelo.

### 3.3. Métricas de Evaluación Definidas para T09 y T10

A partir de esta decisión, se establecen las siguientes métricas formales:

#### Métricas Primarias (Regresión)

| Métrica | Fórmula | Propósito |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum_{k=1}^{n}\lvert y_k - \hat{y}_k \rvert$ | Error promedio en estudiantes afectados |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{k=1}^{n}(y_k - \hat{y}_k)^2}$ | Penaliza errores grandes (outliers masivos) |
| **$R^2$** | $1 - \frac{\sum(y_k - \hat{y}_k)^2}{\sum(y_k - \bar{y})^2}$ | Proporción de varianza explicada por el modelo |

#### Métricas Secundarias (Clasificación Derivada con $\tau = 15$)

| Métrica | Propósito |
|---|---|
| **Accuracy** | Proporción global de aciertos |
| **Precisión** | De los pares marcados como conflicto, ¿cuántos lo son realmente? |
| **Recall** | De los conflictos reales, ¿cuántos detectó el modelo? |
| **F1-Score** | Media armónica de Precisión y Recall |
| **Matriz de Confusión** | Distribución detallada de VP, VN, FP, FN |

> [!IMPORTANT]
> **Priorización del Recall sobre la Precisión:** En el contexto de planificación académica, un **falso negativo** (no detectar un conflicto real → estudiantes perjudicados) es más costoso que un **falso positivo** (marcar un conflicto inexistente → se agrega un turno extra pero nadie pierde una mesa). Por lo tanto, al elegir entre modelos en T09, se priorizará el **Recall** como métrica de desempate cuando el F1-Score sea comparable.

---

## 4. Impacto en los Tickets Posteriores

| Ticket | Implicancia de la decisión |
|---|---|
| **T08 — Pipeline de features** | El target es `cant_estudiantes_afectados` (numérico). Se usará `StandardScaler` o `RobustScaler` para features y no se necesita codificar el target. |
| **T09 — Entrenar y comparar modelos** | Se entrenan regresores (Regresión Lineal, KNN Regressor, Árbol de Decisión Regressor). Se comparan con MAE, RMSE y $R^2$ sobre validación. |
| **T10 — Evaluación final sobre test** | Se reportan métricas de regresión + métricas de clasificación derivada ($\tau = 15$). Se genera gráfico de predicho vs. real y matriz de confusión derivada. |
| **T11 — Análisis de errores** | Se identifican los pares con mayor $\lvert y - \hat{y} \rvert$, analizando si el error se concentra en pares masivos (básicas de 1° año), en la zona cercana al umbral, o en combinaciones inter-carrera. |
| **T12 — Persistencia del modelo** | Se serializa el regresor entrenado con `joblib`. El endpoint de predicción devuelve $\hat{y}_{ij}$ (continuo). |
| **T13 — Construcción del grafo** | Las aristas se crean con peso $w_{ij} = \hat{y}_{ij}$ y se filtran por $\tau$ recibido como parámetro. |
| **T22 — Selector de umbral (Frontend)** | El slider de $\tau$ invoca al endpoint con el nuevo umbral; el backend reconstruye el grafo y devuelve el cronograma actualizado. |
