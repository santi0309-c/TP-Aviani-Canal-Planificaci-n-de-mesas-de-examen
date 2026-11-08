# Análisis de Errores y Diagnóstico del Modelo — Ticket T11

**Proyecto:** Planificación de Mesas de Examen mediante Grafos y Machine Learning  
**Modelo Analizado:** Regresión Lineal (OLS) persistido en `backend/models/modelo.joblib`  
**Conjunto de Prueba:** 117 pares de materias (semilla `42`, estrictamente aislados)  
**Script Diagnóstico:** `backend/app/ml/error_analysis.py`  
**Resultados Serializados:** `backend/models/diagnostico_errores.json`

---

## 1. Motivación y Metodología

El objetivo de esta etapa diagnóstica es **entender dónde y por qué falla el modelo**, superando la visión puramente agregada provista por métricas como el MAE ($1.68$) o el $R^2$ ($0.9915$). Se busca evaluar:
1. Si el modelo presenta sesgo direccional sistemático (tendencia a subestimar o sobreestimar).
2. Si existen patrones de error asociados a la estructura académica de las materias (mismo año curricular, pertenencia a la misma carrera o volumen de matrícula compartida).
3. La naturaleza de los 6 errores de clasificación producidos en el umbral institucional $\tau = 15$ (4 Falsos Negativos y 2 Falsos Positivos).

### Formulación Matemática de los Residuos
Para cada par de materias $i \in \{1, \dots, 117\}$ del conjunto de test:
- **Residuo:** $e_i = y_i - \hat{y}_i$
  - Si $e_i > 0 \implies y_i > \hat{y}_i$: el modelo **subestima** el conflicto real.
  - Si $e_i < 0 \implies y_i < \hat{y}_i$: el modelo **sobreestima** el conflicto real.
- **Error Absoluto:** $|e_i| = |y_i - \hat{y}_i|$

Los identificadores de materias fueron cruzados con `backend/data/materias.json` para facilitar una interpretación pedagógica e institucional con nombres reales.

---

## 2. Diagnóstico de Sesgo Global (Bias)

A nivel de todo el conjunto de prueba, los indicadores de sesgo arrojan:

| Indicador | Valor | Interpretación |
|---|:---:|---|
| **Media de Residuos ($\mu_e$)** | **-0.1856 estudiantes** | Prácticamente nula. El estimador lineal de Mínimos Cuadrados Ordinarios es no sesgado ($E[e] \approx 0$). |
| **Desvío Estándar ($\sigma_e$)** | **2.1313 estudiantes** | Coherente con el desvío del ruido sintético inyectado en el generador ($\mathcal{N}(0, 2.5^2)$). |
| **Pares Subestimados ($y > \hat{y}$)** | **61 pares (52.14%)** | Leve mayoría, indicando que el modelo predijo levemente por debajo en la mitad de los casos. |
| **Pares Sobreestimados ($y < \hat{y}$)** | **56 pares (47.86%)** | Distribución simétrica casi perfecta (52% vs 48%). |
| **Pares con Predicción Exacta** | **0 pares (0.00%)** | Comportamiento esperable en variables continuas generadas por regresión lineal. |

> **Conclusión sobre sesgo:** No se evidencia subestimación ni sobreestimación sistemática global. El modelo distribuye los residuos de forma gaussiana y centrada en cero a lo largo del espectro evaluado.

---

## 3. Detección de Patrones Sistemáticos por Segmentos

### 3.1. Según si pertenecen al mismo año lectivo (`mismo_anio`)

| Condición | Cantidad ($N$) | MAE (Estudiantes) | Desvío Error (Std) |
|---|:---:|:---:|:---:|
| **Distinto Año (`False`)** | 100 | **1.5806** | 1.2782 |
| **Mismo Año (`True`)** | 17 | **2.2735** | 1.5016 |

- **Hallazgo:** Los pares de materias dictadas en el **mismo año de la carrera** presentan un error absoluto medio superior ($2.27$ vs. $1.58$).
- **Causa:** Las materias del mismo año exhiben correlaciones más complejas de cursada y mayor variabilidad intrínseca en la matrícula de alumnos compartidos, acumulando más ruido estocástico.

### 3.2. Según si pertenecen a la misma carrera (`misma_carrera`)

| Condición | Cantidad ($N$) | MAE (Estudiantes) | Desvío Error (Std) |
|---|:---:|:---:|:---:|
| **Distinta Carrera (`False`)** | 30 | **1.1517** | 0.5884 |
| **Misma Carrera (`True`)** | 87 | **1.8639** | 1.4605 |

- **Hallazgo:** En pares inter-carrera el error es notablemente más bajo ($1.15$ alumnos de MAE con desviación de apenas $0.59$).
- **Causa:** La mayor parte de los pares inter-carrera tienen nula o muy baja superposición real de estudiantes, por lo que el modelo predice valores cercanos a cero con gran precisión y mínima dispersión.

### 3.3. Según el volumen de estudiantes compartidos

| Rango de Estudiantes | Cantidad ($N$) | MAE (Estudiantes) | Desvío Error (Std) |
|---|:---:|:---:|:---:|
| **Bajo ($< 15$)** | 65 | **1.2982** | 0.9401 |
| **Medio ($[15, 50)$)** | 37 | **2.0824** | 1.4969 |
| **Alto ($\ge 50$)** | 15 | **2.3520** | 1.7992 |

- **Hallazgo:** El error absoluto escala gradualmente con el volumen de estudiantes compartidos.
- **Causa:** Esto responde a un fenómeno natural de **heterocedasticidad moderada** propia de procesos de recuento poblacional: a mayor masa de alumnos involucrados, la fluctuación absoluta admisible crece, si bien el error porcentual relativo decrece significativamente (un error de 2.3 alumnos sobre 80 estudiantes representa menos del 3% de error relativo).

---

## 4. Top 10 de Pares con Mayor Error Absoluto

A continuación se detallan los 10 pares con mayor discrepancia entre el valor observado y la predicción:

| Par | Materia 1 | Materia 2 | Comp. | Mismo Año | Misma Carrera | Real ($y$) | Pred ($\hat{y}$) | Residuo ($e$) | Error Abs ($|e|$) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **M18-M19** | Sistemas Operativos | Investigación Operativa | 48 | Sí | Sí | 40 | 33.53 | **+6.47** | 6.47 |
| **M05-M11** | Informática General | Probabilidad y Estadística | 75 | No | Sí | 40 | 46.12 | **-6.12** | 6.12 |
| **M16-M20** | Termodinámica Básica | Gestión de Procesos | 32 | No | Sí | 16 | 21.30 | **-5.30** | 5.30 |
| **M03-M05** | Física I | Informática General | 135 | Sí | Sí | 89 | 83.75 | **+5.25** | 5.25 |
| **M30-M37** | Diseño de Plantas Industriales | Simulación de Sist. Productivos | 17 | No | Sí | 17 | 12.64 | **+4.36** | 4.36 |
| **M02-M13** | Análisis Matemático I | Economía General | 56 | No | Sí | 31 | 35.15 | **-4.15** | 4.15 |
| **M09-M18** | Análisis Matemático II | Sistemas Operativos | 33 | No | Sí | 26 | 21.88 | **+4.12** | 4.12 |
| **M21-M40** | Logística y Abastecimiento | Proyecto Final Integrador | 9 | No | Sí | 4 | 8.03 | **-4.03** | 4.03 |
| **M22-M38** | Mecánica de Fluidos | Control de Procesos Químicos | 2 | No | Sí | 0 | 3.99 | **-3.99** | 3.99 |
| **M11-M17** | Probabilidad y Estadística | Arquitectura de Computadores | 35 | No | Sí | 27 | 23.03 | **+3.97** | 3.97 |

---

## 5. Análisis Exhaustivo de los Casos Frontera en $\tau = 15$

Al binarizar las predicciones para armar el grafo de conflictos institucionales, se registraron exactamente **6 clasificaciones erróneas**:

### 5.1. Falsos Negativos ($N=4$): Conflicto Real no Detectado ($y \ge 15 \land \hat{y} < 15$)

| Par | Materia 1 | Materia 2 | Compartidos | Real ($y$) | Pred ($\hat{y}$) | Residuo ($e$) | Margen al Umbral |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **M30-M37** | Diseño de Plantas Industriales | Simulación de Sist. Productivos | 17 | 17 | 12.64 | +4.36 | -2.36 |
| **M15-M29** | Estructuras de Datos | Seguridad e Higiene Industrial | 18 | 16 | 13.22 | +2.78 | -1.78 |
| **M17-M25** | Arquitectura de Computadores | Redes y Telecomunicaciones | 17 | 15 | 12.64 | +2.36 | -2.36 |
| **M02-M18** | Análisis Matemático I | Sistemas Operativos | 19 | 16 | 13.80 | +2.20 | -1.20 |

**Diagnóstico crítico de los FN:**
- Los 4 pares son casos estrictamente **fronterizos**: los valores reales se ubican entre **15 y 17 estudiantes** (apenas superando el umbral de 15), mientras que las predicciones del modelo se sitúan entre **12.64 y 13.80**.
- La discrepancia absoluta promedio de estos 4 casos es de solo **2.92 alumnos**.
- Ningún par con conflicto severo ($y \ge 20$) fue omitido por el modelo.

### 5.2. Falsos Positivos ($N=2$): Alerta de Conflicto Inexistente ($y < 15 \land \hat{y} \ge 15$)

| Par | Materia 1 | Materia 2 | Compartidos | Real ($y$) | Pred ($\hat{y}$) | Residuo ($e$) | Margen al Umbral |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **M08-M22** | Fundamentos de Programación | Mecánica de Fluidos | 24 | 14 | 16.68 | -2.68 | +1.68 |
| **M19-M28** | Investigación Operativa | Control de Calidad | 23 | 14 | 16.11 | -2.11 | +1.11 |

**Diagnóstico crítico de los FP:**
- Ambos pares corresponden exactamente a **14 estudiantes afectados en la realidad**, es decir, se encuentran a una distancia de **1 solo alumno** del límite reglamentario de 15.
- El modelo predijo $16.68$ y $16.11$, sobreestimando por menos de 3 estudiantes.
- Desde una perspectiva operativa, alertar preventivamente sobre pares con 14 estudiantes perjudicados es inocuo para la gestión de mesas (agrega una restricción preventiva de separación horaria sin perjudicar a los alumnos).

---

## 6. Síntesis y Redacción para el Informe Académico

A continuación se transcribe el párrafo formal de conclusiones diagnósticas exigido como criterio de aceptación del ticket:

> ### Diagnóstico de Residuos y Patrones de Error del Modelo
> 
> El análisis de residuos sobre el conjunto de test ($N = 117$) evidencia una distribución prácticamente insesgada ($\mu_e = -0.1856$, $\sigma_e = 2.1313$) con un equilibrio notable entre predicciones subestimadas ($52.14\%$) y sobreestimadas ($47.86\%$). Las mayores discrepancias se concentran en materias del mismo año curricular ($\text{MAE} = 2.27$ vs. $1.58$ en años distintos) y de gran matrícula común, donde la varianza intrínseca del proceso es superior. 
> 
> Al evaluar los 6 desaciertos en la clasificación con umbral $\tau = 15$, se comprueba que todos corresponden a **casos frontera**:
> 1. **Ejemplo 1 (Subestimación / Falso Negativo):** En el par `[M30-M37]` (*Diseño de Plantas Industriales* vs. *Simulación de Sistemas Productivos*), la cantidad real fue de 17 estudiantes afectados mientras que el modelo estimó 12.64 ($e = +4.36$). El modelo subestimó ligeramente el conflicto dejándolo apenas por debajo del umbral, lo que en el algoritmo de coloreo de grafos se traduciría en no agregar la arista de conflicto a menos que se aplique un margen de tolerancia preventivo.
> 2. **Ejemplo 2 (Sobreestimación / Falso Positivo):** En el par `[M08-M22]` (*Fundamentos de Programación* vs. *Mecánica de Fluidos*), con 14 estudiantes reales afectados y 24 inscriptos compartidos, el modelo predijo 16.68 ($e = -2.68$). Esta leve sobreestimación cruzó el umbral clasificándolo como conflicto y generando una falsa alarma; sin embargo, al tratarse de un par con 14 estudiantes (a solo 1 del corte institucional), la decisión es conservadora y no compromete la factibilidad operativa del cronograma de exámenes.
> 3. **Ejemplo 3 (Discrepancia Cuantitativa en Rango Medio):** En el par `[M18-M19]` (*Sistemas Operativos* vs. *Investigación Operativa*), correspondiente al error absoluto máximo del conjunto de test ($6.47$ alumnos), el valor real fue de 40 afectados y el predicho de 33.53. A pesar de este residuo cuantitativo, ambos valores se ubican holgadamente por encima del umbral de 15, por lo que la clasificación cualitativa fue correcta y la arista de conflicto en el grafo se generó sin ambigüedad.

---

## 7. Recomendaciones Operativas para la Construcción del Grafo

A partir del diagnóstico de casos frontera, se recomienda para el algoritmo de grafos:
- **Banda de guarda / Zona de advertencia:** Tratar el intervalo $[12, 15)$ como *"potencial conflicto"* para alertar al planificador académico o permitirle al usuario del backend ajustar el umbral de sensibilidad de forma interactiva (ej. $\tau = 13$).
- **Robustez del modelo lineal:** La ausencia de anomalías groseras o residuos extremos confirma la solidez de la Regresión Lineal regularizada para servir como motor de inferencia en el backend.
