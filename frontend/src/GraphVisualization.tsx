import type { Core, EdgeSingular, ElementDefinition } from 'cytoscape'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { postCronograma } from './api'
import type { CronogramaResponse } from './types'
import { UmbralSelector } from './UmbralSelector'
import { useApiRequest } from './useApiRequest'

interface EdgeDetails {
  source: string
  target: string
  sourceId: string
  targetId: string
  weight: number
}

export interface GraphVisualizationProps {
  initialUmbral?: number
  umbral?: number
  onUmbralChange?: (umbral: number) => void
}

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
  const namesById = new Map(
    cronograma.nodos.map((node) => [node.id, node.nombre]),
  )
  const edges: ElementDefinition[] = cronograma.aristas.map((arista) => ({
    data: {
      id: `edge-${encodeURIComponent(arista.source)}-${encodeURIComponent(arista.target)}`,
      source: arista.source,
      target: arista.target,
      sourceLabel: namesById.get(arista.source) ?? arista.source,
      targetLabel: namesById.get(arista.target) ?? arista.target,
      weight: arista.peso,
      lineWidth: 1 + (Math.sqrt(arista.peso) / Math.sqrt(maxWeight)) * 5,
    },
  }))
  const nodes: ElementDefinition[] = cronograma.nodos.map((nodo) => ({
    data: {
      id: nodo.id,
      label: nodo.nombre,
      color: turnoColor(nodo.turno),
      turno: nodo.turno,
    },
  }))

  return [...edges, ...nodes]
}

function edgeDetails(edge: EdgeSingular): EdgeDetails {
  const sourceLabel: unknown = edge.data('sourceLabel')
  const targetLabel: unknown = edge.data('targetLabel')
  const weight: unknown = edge.data('weight')

  return {
    source: typeof sourceLabel === 'string' ? sourceLabel : edge.source().id(),
    target: typeof targetLabel === 'string' ? targetLabel : edge.target().id(),
    sourceId: edge.source().id(),
    targetId: edge.target().id(),
    weight: typeof weight === 'number' ? weight : 0,
  }
}

function updateGraph(instance: Core, cronograma: CronogramaResponse): void {
  const nodesById = new Map(cronograma.nodos.map((node) => [node.id, node]))
  const elements = buildElements(cronograma)
  const edges = elements.filter((element) => 'source' in element.data)
  const extent = instance.extent()
  const centerX = (extent.x1 + extent.x2) / 2
  const centerY = (extent.y1 + extent.y2) / 2

  instance.batch(() => {
    instance.edges().remove()
    instance.nodes().forEach((node) => {
      const updatedNode = nodesById.get(node.id())
      if (!updatedNode) {
        node.remove()
        return
      }

      node.data({
        label: updatedNode.nombre,
        color: turnoColor(updatedNode.turno),
        turno: updatedNode.turno,
      })
      nodesById.delete(node.id())
    })

    const newNodes: ElementDefinition[] = [...nodesById.values()].map((node, index) => ({
      data: {
        id: node.id,
        label: node.nombre,
        color: turnoColor(node.turno),
        turno: node.turno,
      },
      position: {
        x: centerX + Math.cos(index) * 40,
        y: centerY + Math.sin(index) * 40,
      },
    }))

    instance.add([...newNodes, ...edges])
  })
  instance.resize()
}

export function GraphVisualization({
  initialUmbral = 15,
  umbral: externalUmbral,
  onUmbralChange,
}: GraphVisualizationProps = {}) {
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

  const loadCronograma = useCallback(
    (signal: AbortSignal) => postCronograma({ umbral: activeUmbral }, signal),
    [activeUmbral],
  )

  const { data, loading, error, status, execute } =
    useApiRequest<CronogramaResponse>(loadCronograma)
  const graphElement = useRef<HTMLDivElement>(null)
  const graphInstance = useRef<Core | null>(null)
  const [graphError, setGraphError] = useState<string | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<EdgeDetails | null>(null)

  const invalidColoring = useMemo(
    () => (data ? coloringError(data) : null),
    [data],
  )

  useEffect(() => {
    if (!data || invalidColoring || !graphElement.current || data.nodos.length === 0) {
      return
    }

    let active = true
    if (graphInstance.current && !graphInstance.current.destroyed()) {
      updateGraph(graphInstance.current, data)
    } else {
      void import('cytoscape')
        .then(({ default: cytoscape }) => {
          if (!active || !graphElement.current) {
            return
          }

          const instance = cytoscape({
            container: graphElement.current,
            elements: buildElements(data),
            style: [
              {
                selector: 'node',
                style: {
                  'background-color': 'data(color)',
                  'border-color': '#f8fafc',
                  'border-width': 1,
                  color: '#f8fafc',
                  label: 'data(label)',
                  'font-size': 9,
                  'font-weight': 600,
                  'text-wrap': 'wrap',
                  'text-max-width': '110px',
                  'text-valign': 'center',
                  'text-halign': 'center',
                  width: 110,
                  height: 52,
                  padding: '10px',
                  'text-outline-color': '#101522',
                  'text-outline-width': 2,
                },
              },
              {
                selector: 'edge',
                style: {
                  width: 'data(lineWidth)',
                  'line-color': '#94a3b8',
                  opacity: 0.48,
                  'curve-style': 'bezier',
                },
              },
            ],
            layout: {
              name: 'cose',
              animate: true,
              animationDuration: 500,
              fit: true,
              padding: 40,
              nodeRepulsion: 30000,
              idealEdgeLength: 110,
            },
            minZoom: 0.2,
            maxZoom: 3,
          })

          instance.on('mouseover', 'edge', (event) => {
            setSelectedEdge(edgeDetails(event.target))
          })
          instance.on('mouseout', 'edge', () => setSelectedEdge(null))
          instance.on('tap', 'edge', (event) => {
            setSelectedEdge(edgeDetails(event.target))
          })
          graphInstance.current = instance
        })
        .catch(() => {
          if (active) {
            setGraphError('No se pudo iniciar la visualización interactiva del grafo.')
          }
        })
    }

    return () => {
      active = false
    }
  }, [data, invalidColoring])

  useEffect(
    () => () => {
      graphInstance.current?.destroy()
      graphInstance.current = null
    },
    [],
  )

  const fitGraph = useCallback(() => {
    graphInstance.current?.fit(undefined, 40)
  }, [])

  const turnos = useMemo(
    () =>
      data
        ? [...new Set(data.nodos.map((nodo) => nodo.turno))].sort((a, b) => a - b)
        : [],
    [data],
  )

  const displayedEdge = selectedEdge && data?.aristas.some(
    (edge) =>
      edge.source === selectedEdge.sourceId &&
      edge.target === selectedEdge.targetId &&
      edge.peso === selectedEdge.weight,
  )
    ? selectedEdge
    : null

  const isInitialLoading = loading && !data
  const isUpdatingGraph = loading && Boolean(data)

  return (
    <section className="card glass-card graph-card" aria-labelledby="graph-title">
      <div className="card-header">
        <div>
          <h2 className="status-title" id="graph-title">Grafo de conflictos</h2>
          <p className="graph-subtitle">
            Las materias conectadas tienen turnos diferentes. Arrastrá para explorar y usá la rueda para acercar.
          </p>
        </div>
        {data && (
          <span className="status-pill pill-success">
            {data.cantidad_nodos} materias · {data.cantidad_aristas} conflictos · τ = {data.umbral} ·{' '}
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
              {status !== null && <small>Código HTTP {status}</small>}
            </div>
            <button
              className="btn btn-primary"
              type="button"
              onClick={() => void execute(loadCronograma).catch(() => {})}
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

        {!loading && !error && !invalidColoring && data && data.nodos.length === 0 && (
          <div className="state-container">
            <p>El cronograma no contiene materias para mostrar.</p>
          </div>
        )}

        <div className="graph-viewport-wrapper">
          <div
            className="graph-canvas"
            ref={graphElement}
            style={{
              display:
                data && !invalidColoring && !graphError && data.nodos.length > 0
                  ? undefined
                  : 'none',
            }}
            role="img"
            aria-label={`Grafo con ${data?.cantidad_nodos ?? 0} materias y ${data?.cantidad_aristas ?? 0} conflictos para umbral ${activeUmbral}`}
          />

          {isUpdatingGraph && (
            <div className="graph-updating-overlay" aria-live="polite">
              <span className="btn-spinner" />
              <span>Recalculando para τ = {activeUmbral}...</span>
            </div>
          )}
        </div>

        {!error && !invalidColoring && !graphError && data && data.nodos.length > 0 && (
          <>
            <div className="graph-toolbar">
              <span aria-live="polite">
                {displayedEdge
                  ? `${displayedEdge.source} ↔ ${displayedEdge.target}: ${displayedEdge.weight.toFixed(2)} estudiantes estimados`
                  : 'Pasá el cursor o seleccioná una arista para ver el conflicto exacto.'}
              </span>
              <button
                className="btn btn-primary graph-fit-button"
                type="button"
                onClick={fitGraph}
              >
                Ajustar vista
              </button>
            </div>
            <div className="graph-legend" aria-label="Leyenda de turnos">
              <strong>Turnos ({turnos.length})</strong>
              {turnos.map((turno) => (
                <span className="legend-item" key={turno}>
                  <span
                    className="legend-color"
                    style={{ backgroundColor: turnoColor(turno) }}
                  />
                  Turno {turno + 1}
                </span>
              ))}
              <span className="legend-edge">
                El grosor de la arista indica el peso del conflicto
              </span>
            </div>
          </>
        )}
      </div>
    </section>
  )
}
