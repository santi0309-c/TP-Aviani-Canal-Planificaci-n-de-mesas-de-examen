# Comparación y Selección de Modelos Predictivos — Ticket T09

**Proyecto:** Planificación de Mesas de Examen mediante Grafos y Machine Learning  
**Módulo:** `backend/app/ml/train.py`  
**Dataset:** `backend/data/pares_materias.csv` (780 pares, partición 70% Train / 15% Validación / 15% Test)  
**Target:** `cant_estudiantes_afectados` (Regresión continua)

---

## 1. Metodología de Entrenamiento y Validación

Para resolver el problema de estimación del volumen de conflicto entre materias, se evaluaron **tres familias algorítmicas clásicas** vistas en el curso:
1. **Regresión Lineal:** Modelo paramétrico lineal (Mínimos Cuadrados Ordinarios).
2. **Árbol de Decisión Regresor (`DecisionTreeRegressor`):** Modelo no lineal basado en particiones ortogonales sucesivas.
3. **K-Nearest Neighbors Regresor (`KNeighborsRegressor`):** Modelo no paramétrico basado en distancias en el espacio de features escaladas.

### 1.1. Garantía de Aislamiento (Prevención de Fuga de Datos)
- El ajuste de hiperparámetros se realizó **exclusivamente sobre el conjunto de Train (545 pares)** mediante **Validación Cruzada de 5 Pliegues (5-Fold Cross-Validation)** estratificada por bloques con semilla fija `42`.
- El conjunto de **Validación (118 pares)** se utilizó exclusivamente para comparar el desempeño final de los mejores estimadores y seleccionar el modelo ganador.
- El conjunto de **Test (117 pares)** se mantuvo **estrictamente intacto**, garantizando que ninguna decisión de modelado ni ajuste de parámetros estuviera condicionada por los datos de prueba final (reservados para T10).

---

## 2. Ajuste de Hiperparámetros con 5-Fold CV

Para cada familia se utilizó `GridSearchCV` sobre Train con métrica de optimización de error cuadrático (`neg_root_mean_squared_error`):

| Modelo | Hiperparámetros Explorados | Configuración Óptima Encontrada |
|---|---|---|
| **Regresión Lineal** | Estimación OLS sin hiperparámetros libres | Coeficientes analíticos cerrados |
| **Árbol de Decisión** | • `max_depth`: `[3, 5, 8, 12, None]`<br>• `min_samples_split`: `[2, 5, 10]`<br>• `min_samples_leaf`: `[1, 2, 4]` | `max_depth = 12`<br>`min_samples_split = 2`<br>`min_samples_leaf = 2` |
| **KNN** | • `n_neighbors`: `[3, 5, 7, 9, 11, 15]`<br>• `weights`: `['uniform', 'distance']`<br>• `p`: `[1 (Manhattan), 2 (Euclídea)]` | `n_neighbors = 5`<br>`weights = 'distance'`<br>`p = 1 (Manhattan)` |

---

## 3. Tabla Comparativa sobre el Conjunto de Validación

Evaluación de los tres modelos optimizados sobre el conjunto de Validación (118 pares no vistos durante el ajuste de hiperparámetros):

| Modelo | CV RMSE (Train) | Val MAE (alumnos) | Val RMSE (alumnos) | Val $R^2$ | Train $R^2$ | Brecha $R^2$ (Sobreajuste) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🏆 **Regresión Lineal** | **2.275** | **1.629** | **2.124** | **0.9913** | **0.9905** | **-0.0008** (Generalización ideal) |
| **Árbol de Decisión** | 2.338 | 1.761 | 2.552 | 0.9874 | 0.9945 | +0.0071 (Leve sobreajuste) |
| **KNN** | 2.339 | 1.683 | 2.467 | 0.9882 | 0.9950 | +0.0068 (Leve sobreajuste) |

> [!NOTE]
> - **Val MAE:** Error promedio absoluto expresado en cantidad de estudiantes afectados.
> - **Val RMSE:** Raíz del error cuadrático medio, penaliza desvíos grandes.
> - **Val $R^2$:** Proporción de la varianza explicada por el modelo sobre datos nuevos.

---

## 4. Análisis Crítico de los Modelos

### 4.1. Regresión Lineal (Modelo Ganador)
* **Desempeño sobresaliente:** Logró el menor error absoluto medio ($\text{MAE} = 1.629$ estudiantes) y el menor error cuadrático ($\text{RMSE} = 2.124$ estudiantes), explicando el **99.13%** de la varianza del target en validación.
* **Excelente equilibrio Sesgo / Varianza:** El $R^2$ en Train ($0.9905$) y Validación ($0.9913$) es prácticamente idéntico. No sufre de sobreajuste ni de dispersión errática ante extrapolaciones.
* **Interpretabilidad total:** Los coeficientes estimados reflejan directamente la física y las reglas académicas del problema (porción directa de compartidos más término aditivo por año y carrera).

### 4.2. Árbol de Decisión
* Si bien alcanzó un $R^2 = 0.9874$, presentó un $\text{RMSE} = 2.552$ (un 20% superior al modelo lineal).
* Los árboles realizan particiones ortogonales por umbrales duros, lo que genera predicciones escalonadas y constantes a tramos. Esto introduce un error de discretización inevitable en relaciones lineales continuas.
* Se observa un leve sobreajuste en Train ($R^2 = 0.9945$ vs $0.9874$ en validación).

### 4.3. K-Nearest Neighbors (KNN)
* Con $k=5$ y ponderación por distancia inversa (`weights='distance'`), mejoró respecto al árbol ($\text{MAE} = 1.683$, $R^2 = 0.9882$).
* Sin embargo, es más sensible a la densidad local de pares en los extremos de la distribución (materias masivas de 1° año vs. materias electivas pequeñas de 5° año) y requiere mantener todo el dataset de entrenamiento en memoria para calcular distancias en cada inferencia.

---

## 5. Decisión Final y Modelo Seleccionado

Se elige la **Regresión Lineal** como el modelo productivo definitivo para ser persistido en `models/modelo.joblib`.

### Justificación de la Elección:
1. **Superioridad empírica:** Supera a todos los competidores en las tres métricas evaluadas sobre el conjunto de validación (menor MAE, menor RMSE, mayor $R^2$).
2. **Principio de Parsimonia (Navaja de Ockham):** Entre modelos con rendimiento comparable o superior, se debe preferir siempre el modelo más simple. La Regresión Lineal tiene costo de inferencia $\mathcal{O}(d)$ instantáneo (3 multiplicaciones y 1 suma), facilitando predicciones en tiempo real dentro de la API FastAPI.
3. **Coherencia con el Dominio Académico:** Como se analizó en el EDA (T06), la cantidad de estudiantes afectados responde a una combinación aditiva proporcional de alumnos compartidos con desplazamientos constantes por año y carrera. Un modelo lineal captura esta estructura sin sufrir artefactos de discontinuidad.

---

## 6. Persistencia y Exportación
El pipeline completo ganador (`preprocessor` + `LinearRegression`) fue exportado mediante `joblib`:
* **Archivo serializado:** `backend/models/modelo.joblib`
* **Entrada esperada:** DataFrame con las columnas `cant_estudiantes_compartidos_historico`, `mismo_anio`, `misma_carrera`.
* **Salida producida:** Estimación continua $\hat{y}$ (alumnos afectados).
