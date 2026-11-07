# Preprocesamiento y Partición de Datos — Ticket T08

**Proyecto:** Planificación de Mesas de Examen mediante Grafos y Machine Learning  
**Módulo:** `backend/app/ml/features.py`  
**Dataset:** `backend/data/pares_materias.csv` (780 pares, $\binom{40}{2}$)

---

## 1. Selección de Features

### 1.1. Features Predictoras ($X$)

| # | Feature | Tipo | Transformación | Justificación |
|---|---|---|---|---|
| 1 | `cant_estudiantes_compartidos_historico` | `int` (numérica) | `StandardScaler` | Variable predictora principal: cantidad de estudiantes que cursaron ambas materias. Presenta asimetría positiva (mediana = 13, máx = 205), por lo que requiere escalado. |
| 2 | `mismo_anio` | `bool` → `int` | Passthrough {0, 1} | Indicador categórico binario de co-pertenencia al mismo año lectivo. Ya codificado numéricamente, no requiere escalado. |
| 3 | `misma_carrera` | `bool` → `int` | Passthrough {0, 1} | Indicador categórico binario de pertenencia a la misma carrera o tronco común. Mismo tratamiento que `mismo_anio`. |

### 1.2. Columnas Excluidas

| Columna Excluida | Motivo |
|---|---|
| `materia_id_1`, `materia_id_2` | Son identificadores institucionales arbitrarios (`M01`, `M02`, ...). Si se incluyen, el modelo **memorizaría** los pares específicos del dataset de entrenamiento en lugar de aprender patrones generalizables. |
| `hubo_conflicto` | Es una **binarización directa del target** ($\mathbb{I}[\text{afectados} \ge 15]$). Incluirla constituiría un caso de **Target Leakage** crítico: el modelo accedería a información derivada de la variable que debe predecir, inflando artificialmente las métricas sin capacidad de generalización real. |

### 1.3. Variable Objetivo ($y$)

$$y = \texttt{cant\_estudiantes\_afectados} \in \mathbb{Z}_{\ge 0}$$

Según la decisión tomada en T07, se utiliza el target continuo de regresión.

---

## 2. Pipeline de Preprocesamiento

Se utiliza un `ColumnTransformer` de `scikit-learn` que aplica transformaciones diferenciadas según el tipo de variable:

```
ColumnTransformer
├── "num": StandardScaler() → [cant_estudiantes_compartidos_historico]
└── "bool": passthrough     → [mismo_anio, misma_carrera]
```

### ¿Por qué `StandardScaler`?

El EDA (T06) reveló que `cant_estudiantes_compartidos_historico` tiene una distribución fuertemente asimétrica con media $= 26.81$ y desvío $= 38.11$. Modelos sensibles a la escala (como KNN o SVM) se benefician del centrado y normalización:

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

> [!IMPORTANT]
> **Prevención de Data Leakage:** El `StandardScaler` se ajusta (`fit`) **exclusivamente** con los datos de Train. Los conjuntos de Validación y Test solo reciben `transform`, utilizando la media ($\mu_{\text{train}}$) y el desvío ($\sigma_{\text{train}}$) calculados sobre Train. Esto garantiza que ninguna estadística de los datos de evaluación se filtre al proceso de entrenamiento.

---

## 3. Partición de Datos

### 3.1. Esquema de Partición

| Conjunto | Proporción | Filas (N=780) | Uso |
|---|:---:|:---:|---|
| **Train** | 70% | 546 | Ajustar los modelos y el preprocesador |
| **Validación** | 15% | 117 | Comparar modelos y ajustar hiperparámetros (T09) |
| **Test** | 15% | 117 | Evaluación final única e irrevocable (T10) |

### 3.2. Implementación

La partición se realiza en dos etapas secuenciales con `train_test_split`:

1. **Etapa 1:** Separar Test (15%) del resto (85%).
2. **Etapa 2:** Separar Validación ($\frac{0.15}{0.85} \approx 17.65\%$ del resto) de Train (70% del total).

Ambas etapas utilizan `random_state=42` para **reproducibilidad determinista**: ejecutar el pipeline cualquier cantidad de veces produce exactamente las mismas particiones.

### 3.3. Garantías Metodológicas

- **Sin solapamiento:** La intersección entre los índices de Train, Validación y Test es vacía ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- **Cobertura total:** La unión de los tres conjuntos cubre las 780 filas del dataset.
- **Test aislado:** El conjunto de Test **no participa** en ninguna decisión de entrenamiento, selección de hiperparámetros ni comparación de modelos. Solo se evalúa una única vez en T10.

> [!NOTE]
> Los metadatos de la partición (semilla, proporciones, nombres de columnas) se pueden exportar a un archivo JSON mediante `save_split_metadata()`, garantizando la trazabilidad completa del experimento.

