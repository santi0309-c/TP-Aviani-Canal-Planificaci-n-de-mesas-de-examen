# Sección de Uso de Asistentes de Inteligencia Artificial

---

## 1. Introducción y Criterio General

Durante el desarrollo del Trabajo Práctico utilizamos asistentes de inteligencia artificial generativa como herramientas complementarias de asistencia al desarrollo, documentación y consulta técnica. Desde el inicio acordamos una pauta clara: **la IA funciona como acelerador de escritura y consultor de sintaxis, pero nunca como sustituto del razonamiento algorítmico, el modelado matemático ni las decisiones de arquitectura del sistema**.

Este documento recopila de forma honesta, transparente y con casos reales cómo se integraron estas herramientas en el flujo de trabajo de nuestro equipo.

---

## 2. Herramientas Utilizadas

| Herramienta | Proveedor / Entorno | Uso Principal |
|---|---|---|
| **Claude (Claude 3.5 Sonnet)** | Anthropic (interfaz web y API) | Consultas arquitectónicas, asistencia en la redacción de informes técnicos, diseño de estructuras de tests y diagnóstico de errores complejos. |
| **GitHub Copilot** | GitHub / Microsoft (extensión para VS Code) | Autocompletado contextual en línea, sugerencia de firmas de funciones, tipado en TypeScript y boilerplate de pruebas unitarias. |
| **ChatGPT (GPT-4o)** | OpenAI (interfaz web) | Consultas rápidas sobre sintaxis puntual de bibliotecas (NetworkX, Cytoscape.js) y soporte con estilos CSS / Flexbox. |

---

## 3. Dónde SÍ y Dónde NO se utilizó IA

### Dónde SÍ se utilizó:
1. **Scaffolding y boilerplate:**
   - Estructura base de los schemas de Pydantic v2 en `backend/app/schemas.py`.
   - Componentes esqueleto en React con TypeScript (configuración inicial de Vite, imports de Cytoscape.js).
2. **Generación y parametrización de tests:**
   - Creación de combinaciones para `@pytest.mark.parametrize` probando distintas familias de grafos sintéticos con NetworkX (grafos de Erdős–Rényi, Barabási–Albert y Watts–Strogatz).
3. **Estilos y maquetación web:**
   - Ajustes de CSS responsivo, variables de color en modo oscuro, centrado de tarjetas con Flexbox/Grid y diseño de badges para el cronograma.
4. **Revisión de redacción y consistencia:**
   - Detección de inconsistencias entre la documentación y el código fuente; sugerencias para mejorar la claridad expositiva en los informes en Markdown.

### Dónde NO se utilizó:
1. **Formulación matemática del problema:**
   - La modelización del problema de mesas de examen como un problema de coloreo de grafos de conflictos ponderados ($G = (V, E, w)$) donde los vértices son asignaturas y las aristas representan el número esperado de estudiantes afectados fue concebida y diseñada íntegramente por los integrantes.
2. **Implementación del algoritmo de coloreo greedy (Welsh-Powell):**
   - El ordenamiento por grado decreciente ($\Delta(G)$) con desempate por peso acumulado de aristas incidentes y asignación determinística del menor turno entero no negativo disponible ($t \ge 0$) fue razonado y programado a mano para asegurar el cumplimiento estricto de las cotas teóricas vistas en la cátedra (Cota de Brooks).
3. **Decisión de modelado de ML (Regresión vs. Clasificación):**
   - La decisión crítica de entrenar un modelo de regresión continua que estime la cantidad de alumnos afectados ($y \in \mathbb{R}^+$) en lugar de un clasificador binario rígido fue tomada por nosotros para permitir que el umbral de corte $\tau$ sea configurable dinámicamente en tiempo de ejecución por el usuario en el frontend.
4. **Validación formal de invariantes:**
   - La lógica que verifica formalmente que ninguna arista conecte vértices del mismo color ($\forall (u, v) \in E \implies \text{color}(u) \ne \text{color}(v)$) y que el número de turnos sea válido.

---

## 4. Ejemplos Concretos de Prompts Útiles

A continuación transcribimos ejemplos reales de consignas que nos resultaron útiles durante el proyecto:

### Ejemplo 1: Parametrización de pruebas sobre grafos con NetworkX
* **Contexto:** En el ticket T26 necesitábamos verificar que el validador de coloreo no fallara ante grafos aleatorios de diferentes topologías y tamaños.
* **Prompt:**
  > *"Necesito escribir un test unitario en pytest usando `@pytest.mark.parametrize` para probar nuestra función `greedy_coloring(G)`. Queremos generar grafos sintéticos con NetworkX usando semillas fijas (reproducibles): grafos Erdős-Rényi variando n entre 10 y 40 con p entre 0.1 y 0.5, y grafos Barabási-Albert. El test debe verificar que `validate_coloring(G, coloring)` devuelva True y que la cantidad de turnos no supere `max(grados) + 1`."*
* **Resultado:** La IA propuso una estructura limpia de fixtures con semillas determinísticas que sirvió de base para `backend/tests/test_properties.py`.

### Ejemplo 2: Implementación de debounce para el selector de umbral
* **Contexto:** En el frontend (T22), al mover el slider de umbral $\tau$, no queríamos saturar el backend con peticiones en cada pixel de desplazamiento, pero necesitábamos que la interfaz mostrara el valor numérico en tiempo real.
* **Prompt:**
  > *"En React 19 con TypeScript, ¿cómo armo un custom hook o callback de debounce que mantenga un estado local sincronizado inmediatamente con el slider `<input type='range'>` para que se sienta fluido a 60 FPS, pero demore 350 ms antes de llamar a la función `onChange` que dispara el POST a la API?"*
* **Resultado:** Nos orientó en la estructura del hook `useDebounce.ts`, permitiendo separar el valor inmediato visual del valor propagado a la red.

### Ejemplo 3: Manejo de errores HTTP en Pydantic y FastAPI
* **Contexto:** Necesitábamos que el endpoint `/cronograma` respondiera con un error `400 Bad Request` bien formateado si el umbral enviado era negativo.
* **Prompt:**
  > *"En FastAPI con Pydantic v2, ¿cuál es la forma canónica de validar que un campo `umbral: float` en el body de una petición sea mayor o igual a 0 usando `Field`, devolviendo un mensaje explicativo claro en español en caso de fallo?"*
* **Resultado:** Nos sugirió usar `Field(..., ge=0.0, description="Umbral de corte de estudiantes afectados")`, lo cual integró automáticamente la validación con la documentación interactiva de OpenAPI en `/docs`.

---

## 5. Casos en que la IA se Equivocó y Cómo los Detectamos

El uso de asistentes requiere una actitud crítica permanente. Durante el desarrollo encontramos varios errores y alucinaciones técnicas:

### Caso 1: Afirmar que el coloreo greedy es "óptimo" (Error conceptual algorítmico)
* **Qué ocurrió:** Al pedirle que generara un borrador de docstring para la función `greedy_coloring`, el modelo redactó:
  > *"...asigna el cronograma utilizando la menor cantidad óptima posible de turnos de examen (número cromático)."*
* **Cómo lo detectamos:** Inmediatamente nos dimos cuenta del error teórico. Por los contenidos de la materia sabemos que el problema del coloreo óptimo de grafos (encontrar el número cromático $\chi(G)$) es un problema **NP-completo** (Karp, 1972). El algoritmo goloso de Welsh-Powell es una heurística constructiva eficiente de orden polinomial ($O(|V|^2 + |E|)$), pero **no garantiza la solución óptima** (salvo en familias muy particulares de grafos).
* **Solución aplicada:** Reescribimos la documentación aclarando explícitamente que se trata de una aproximación heurística factible que acota el número de turnos a lo sumo en $\Delta(G) + 1$, pero sin asegurar minimalidad cromática.

### Caso 2: Métodos inexistentes en la API de Cytoscape.js (Alucinación de biblioteca)
* **Qué ocurrió:** Al intentar refrescar el grosor de las aristas sin resetear las coordenadas de los nodos cuando el usuario cambiaba el umbral, Copilot sugirió:
  ```typescript
  cy.elements().updateData({ weight: newWeight })
  cy.layout({ name: 'preset', animate: 'during' }).run()
  ```
* **Cómo lo detectamos:** TypeScript arrojó un error de compilación (`Property 'updateData' does not exist on type 'Collection'`) y al ejecutarlo Vite falló.
* **Solución aplicada:** Consultamos la documentación oficial de Cytoscape.js v3 y reescribimos la función usando `cy.batch()` iterando sobre los elementos con `.data('peso', nuevoPeso)`, evitando recrear el layout y manteniendo fijas las posiciones de las asignaturas.

### Caso 3: Proponer clasificación binaria para un problema dinámico (Mala elección arquitectónica)
* **Qué ocurrió:** En las primeras etapas del proyecto, al consultar sobre cómo modelar la predicción de conflictos, la IA sugirió entrenar un clasificador binario (`RandomForestClassifier`) que predijera directamente `conflicto = 1` o `0`.
* **Cómo lo detectamos:** Nos dimos cuenta de que esa propuesta rompía el requerimiento central del trabajo práctico: si el modelo predice una clase binaria fija a partir de un umbral rígido predefinido (por ejemplo 15 alumnos), la secretaría académica pierde toda capacidad de regular el umbral en tiempo real desde la aplicación web para evaluar escenarios más o menos restrictivos.
* **Solución aplicada:** Descartamos la sugerencia de la IA y optamos por un modelo de **regresión continua** ($y = \text{estudiantes afectados}$). De esta manera, el modelo estima la magnitud del conflicto una sola vez y el umbral $\tau$ se aplica como un filtro flexible sobre las aristas del grafo en runtime.

---

## 6. Reflexión y Balance Crítico

### En qué ayudó:
- **Reducción de tiempos de escritura:** Ahorró horas de trabajo mecánico escribiendo esquemas repetitivos, tipos auxiliares de TypeScript, fixtures de pruebas y comandos de shell.
- **Exploración de librerías:** Facilitó encontrar rápidamente sintaxis específica de NetworkX y configuraciones de Vite sin tener que leer manuales completos para tareas triviales.
- **Aprendizaje interactivo:** Actuó como un interlocutor rápido para contrastar ideas de implementación de componentes y estructura de carpetas.

### En qué NO ayudó:
- **Criterio algorítmico y matemático:** La IA tiende a ofrecer respuestas estándar que suenan convincentes pero que a menudo violan sutilezas teóricas (como llamar "óptimo" a un algoritmo voraz) o no se adaptan a las restricciones de la cátedra.
- **Coherencia global del sistema:** Cada prompt se responde de forma aislada; la IA no "entiende" la totalidad del pipeline (dataset $\to$ regresión $\to$ filtrado $\to$ grafo $\to$ coloreo $\to$ frontend). La integración y el diseño de extremo a extremo requirieron la visión completa de los autores.

