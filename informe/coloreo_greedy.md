# Coloreo Greedy

<!-- Informe de la tarea T14 (Coloreo greedy) -->

## 5. Resultados visuales

![Visualización del Grafo de Conflictos](capturas/grafo_conflictos_preview.png)

### Impacto del Umbral de Conflicto en el Frontend

A continuación se muestra el comportamiento de la interfaz interactiva y el efecto del umbral de corte ($\tau$) sobre la densidad del grafo y la cantidad de turnos requeridos:

#### 1. Umbral Bajo ($\tau = 5$) — Restricción Alta
Se detectan conflictos incluso con pocos estudiantes en común. El grafo es denso (567 aristas) y requiere 22 turnos de examen para evitar cualquier superposición.

![Umbral bajo (tau=5)](capturas/frontend/01_umbral_bajo_tau5.png)

#### 2. Umbral Recomendado ($\tau = 15$) — Punto de Equilibrio
Punto de equilibrio sugerido por el modelo. Se filtran cruces menores tolerables, resultando en 275 aristas de conflicto y 18 turnos de examen.

![Umbral medio (tau=15)](capturas/frontend/02_umbral_medio_tau15.png)

#### 3. Umbral Alto ($\tau = 40$) — Restricción Baja
Solo se consideran conflictos severos entre materias muy concurrentes. El grafo queda disperso (89 aristas) y el calendario se compacta a 8 turnos.

![Umbral alto (tau=40)](capturas/frontend/03_umbral_alto_tau40.png)

#### 4. Vista de Cronograma (`CronogramaGrid`)
Distribución de las 40 materias en columnas por turno de examen, sincronizadas con la paleta de colores del grafo y buscador de asignaturas.

![Vista de cronograma](capturas/frontend/04_vista_cronograma.png)


