import type { Core, ElementDefinition, LayoutOptions } from 'cytoscape'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { postCronograma } from './api'
import type { CronogramaResponse } from './types'
import { UmbralSelector } from './UmbralSelector'
import { useApiRequest } from './useApiRequest'

function turnoColor(turno: number): string {
  const hue = (turno * 137.508) % 360
  const chroma = (1 - Math.abs(2 * 0.48 - 1)) * 0.72
  const hueSection = hue / 60
  const secondary = chroma * (1 - Math.abs((hueSection % 2) - 1))
  const rgb =
    hueSection < 1
      ? [chroma, secondary, 0]
      : hueSection < 2
        ? [secondary, chroma, 0]
        : hueSection < 3
          ? [0, chroma, secondary]
          : hueSection < 4
            ? [0, secondary, chroma]
            : hueSection < 5
              ? [secondary, 0, chroma]
              : [chroma, 0, secondary]
  const lightnessOffset = 0.48 - chroma / 2
  const channels = rgb.map((channel) =>
    Math.round((channel + lightnessOffset) * 255)
      .toString(16)
      .padStart(2, '0'),
  )

  return `#${channels.join('')}`
}

function coloringError(cronograma: CronogramaResponse): string | null {
  if (!cronograma.es_valido || cronograma.es_coloreo_valido === false) {
    return 'El backend devolvió un coloreo marcado como inválido.'
  }

  const turnosPorMateria = new Map(
    cronograma.nodos.map((nodo) => [nodo.id, nodo.turno]),
  )

  for (const arista of cronograma.aristas) {
    const sourceTurno = turnosPorMateria.get(arista.source)
    const targetTurno = turnosPorMateria.get(arista.target)

    if (sourceTurno === undefined || targetTurno === undefined) {
      return 'El grafo contiene una arista que referencia una materia inexistente.'
    }

    if (sourceTurno === targetTurno) {
      return `El coloreo asignó el mismo turno a materias conectadas (${arista.source} y ${arista.target}).`
    }
  }

  return null
}

function buildElements(cronograma: CronogramaResponse): ElementDefinition[] {
  const maxWeight = Math.max(1, ...cronograma.aristas.map((arista) => arista.peso))
  const nodes: ElementDefinition[] = cronograma.nodos.map((nodo) => ({
    data: {
      id: nodo.id,
      label: nodo.nombre,
      color: turnoColor(nodo.turno),
      turno: nodo.turno,
    },
  }))
  const edges: ElementDefinition[] = cronograma.aristas.map((arista, index) => ({
    data: {
      id: `edge-${index}`,
      source: arista.source,
      target: arista.target,
      peso: Math.round(arista.peso * 10) / 10,
      lineWidth: 1 + (Math.sqrt(arista.peso) / Math.sqrt(maxWeight)) * 5,
    },
  }))

  return [...edges, ...nodes]
}

const LAYOUT_OPTIONS: LayoutOptions = {
  name: 'cose',
  animate: true,
  animationDuration: 400,
  fit: true,
  padding: 35,
  nodeRepulsion: () => 45000,
  idealEdgeLength: () => 100,
  gravity: 0.25,
}

export interface GraphVisualizationProps {
  initialUmbral?: number
  umbral?: number
  onUmbralChange?: (umbral: number) => void
}

export function GraphVisualization({
  initialUmbral = 15,
  umbral: externalUmbral,
  onUmbralChange,
}: GraphVisualizationProps) {
  // Manejo de estado controlado / no controlado para el umbral
  const [internalUmbral, setInternalUmbral] = useState<number>(initialUmbral)
  const activeUmbral = externalUmbral !== undefined ? externalUmbral : internalUmbral

  const handleUmbralChange = useCallback(
    (newUmbral: number) => {
      if (externalUmbral === undefined) {
        setInternalUmbral(newUmbral)
      }
      onUmbralChange?.(newUmbral)
    },
    [externalUmbral, onUmbralChange],
  )

  const { data, loading, error, execute } = useApiRequest<CronogramaResponse>()
  const graphElement = useRef<HTMLDivElement>(null)
  const cyInstanceRef = useRef<Core | null>(null)
  const [graphError, setGraphError] = useState<string | null>(null)

  // Cargar cronograma cada vez que cambie el umbral estabilizado
  useEffect(() => {
    void execute((signal) => postCronograma({ umbral: activeUmbral }, signal)).catch(() => {
      // El estado del hook useApiRequest expone los errores
    })
  }, [activeUmbral, execute])

  const invalidColoring = useMemo(
    () => (data ? coloringError(data) : null),
    [data],
  )

  // Inicialización o actualización reactiva de Cytoscape sin desmontar el canvas
  useEffect(() => {
    if (!data || invalidColoring || !graphElement.current || data.nodos.length === 0) {
      return
    }

    let active = true

    // Si ya existe una instancia de Cytoscape, actualizar elementos de forma fluida
    if (cyInstanceRef.current && !cyInstanceRef.current.destroyed()) {
      const cy = cyInstanceRef.current
      cy.elements().remove()
      cy.add(buildElements(data))
      const layout = cy.layout(LAYOUT_OPTIONS)
      layout.run()
      return
    }

    // Inicialización por primera vez con carga diferida
    void import('cytoscape')
      .then(({ default: cytoscape }) => {
        if (!active || !graphElement.current) {
          return
        }

        const cy = cytoscape({
          container: graphElement.current,
          elements: buildElements(data),
          style: [
            {
              selector: 'node',
              style: {
                'background-color': 'data(color)',
                'border-color': '#f8fafc',
                'border-width': 1.5,
                color: '#f8fafc',
                label: 'data(label)',
                'font-size': 9,
                'font-weight': 600,
                'text-wrap': 'wrap',
                'text-max-width': '100px',
                'text-valign': 'center',
                'text-halign': 'center',
                width: 105,
                height: 50,
                padding: '8px',
                'text-outline-color': '#0c0e14',
                'text-outline-width': 2.5,
              },
            },
            {
              selector: 'edge',
              style: {
                width: 'data(lineWidth)',
                'line-color': '#94a3b8',
                opacity: 0.45,
                'curve-style': 'bezier',
              },
            },
            {
              selector: 'node:selected',
              style: {
                'border-color': '#38bdf8',
                'border-width': 3,
                'text-outline-color': '#0284c7',
              },
            },
            {
              selector: 'edge:selected',
              style: {
                'line-color': '#38bdf8',
                opacity: 0.9,
                width: 4,
              },
            },
          ],
          layout: LAYOUT_OPTIONS,
          minZoom: 0.2,
          maxZoom: 3,
        })

        cyInstanceRef.current = cy
      })
      .catch(() => {
        if (active) {
          setGraphError('No se pudo inicializar el motor de grafos Cytoscape.')
        }
      })

    return () => {
      active = false
    }
  }, [data, invalidColoring])

  // Cleanup final al desmontar el componente
  useEffect(() => {
    return () => {
      cyInstanceRef.current?.destroy()
      cyInstanceRef.current = null
    }
  }, [])

  const handleFitView = useCallback(() => {
    if (cyInstanceRef.current && !cyInstanceRef.current.destroyed()) {
      cyInstanceRef.current.fit(undefined, 40)
    }
  }, [])

  const turnos = useMemo(
    () =>
      data
        ? [...new Set(data.nodos.map((nodo) => nodo.turno))].sort((a, b) => a - b)
        : [],
    [data],
  )

  const isInitialLoading = loading && !data
  const isUpdatingGraph = loading && Boolean(data)

  return (
    <section className="card glass-card graph-card" aria-labelledby="graph-title">
      <div className="card-header">
        <div>
          <h2 className="status-title" id="graph-title">Grafo de conflictos</h2>
          <p className="graph-subtitle">
            Cada nodo representa una materia y su color indica el turno de examen asignado.
            Las materias adyacentes no comparten turno.
          </p>
        </div>
        {data && (
          <span className="status-pill pill-success">
            {data.cantidad_nodos} materias · {data.cantidad_aristas} aristas ·{' '}
            <strong>{data.cantidad_turnos} turnos</strong>
          </span>
        )}
      </div>

      <div className="card-body">
        {/* Selector de Umbral Interactivo con Slider y Presets */}
        <UmbralSelector
          value={activeUmbral}
          onChange={handleUmbralChange}
          disabled={isInitialLoading}
          isUpdating={isUpdatingGraph}
        />

        {isInitialLoading && (
          <div className="state-container state-loading" role="status">
            <div className="spinner" />
            <p>Generando el grafo de conflictos inicial...</p>
          </div>
        )}

        {!loading && error && (
          <div className="state-container state-error" role="alert">
            <div className="state-icon error-icon">✕</div>
            <div className="state-details">
              <h3>No se pudo cargar el grafo</h3>
              <p>{error}</p>
            </div>
            <button
              className="btn btn-primary"
              type="button"
              onClick={() =>
                void execute((signal) =>
                  postCronograma({ umbral: activeUmbral }, signal),
                ).catch(() => {
                  // Error administrado por useApiRequest
                })
              }
            >
              Reintentar
            </button>
          </div>
        )}

        {!loading && !error && invalidColoring && (
          <div className="state-container state-error" role="alert">
            <div className="state-icon error-icon">✕</div>
            <div className="state-details">
              <h3>Coloreo inválido</h3>
              <p>{invalidColoring}</p>
            </div>
          </div>
        )}

        {!loading && !error && !invalidColoring && graphError && (
          <div className="state-container state-error" role="alert">
            <div className="state-icon error-icon">✕</div>
            <div className="state-details">
              <h3>No se pudo mostrar el grafo</h3>
              <p>{graphError}</p>
            </div>
          </div>
        )}

        {!isInitialLoading && !error && !invalidColoring && !graphError && data && (
          <>
            <div className="graph-viewport-wrapper">
              <div
                className="graph-canvas"
                ref={graphElement}
                role="img"
                aria-label={`Grafo con ${data.cantidad_nodos} materias y ${data.cantidad_aristas} conflictos activos para umbral ${activeUmbral}`}
              />

              {isUpdatingGraph && (
                <div className="graph-updating-overlay" aria-live="polite">
                  <span className="btn-spinner" />
                  <span>Recalculando para τ = {activeUmbral}...</span>
                </div>
              )}

              <div className="graph-actions-bar">
                <button
                  type="button"
                  className="btn-graph-action"
                  onClick={handleFitView}
                  title="Ajustar y centrar la vista del grafo"
                >
                  <svg
                    className="action-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3" />
                  </svg>
                  Ajustar vista
                </button>
              </div>
            </div>

            <div className="graph-legend" aria-label="Leyenda de turnos">
              <div className="legend-title-wrapper">
                <strong>Turnos ({turnos.length}):</strong>
                <span className="legend-edge">
                  El grosor de la arista indica el peso del conflicto
                </span>
              </div>
              <div className="legend-items-grid">
                {turnos.map((turno) => (
                  <span className="legend-item" key={turno}>
                    <span
                      className="legend-color"
                      style={{ backgroundColor: turnoColor(turno) }}
                    />
                    Turno {turno + 1}
                  </span>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </section>
  )
}
