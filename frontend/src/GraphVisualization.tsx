import type { Core, ElementDefinition } from 'cytoscape'
import { useEffect, useMemo, useRef, useState } from 'react'
import { postCronograma } from './api'
import type { CronogramaResponse } from './types'
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
    },
  }))
  const edges: ElementDefinition[] = cronograma.aristas.map((arista, index) => ({
    data: {
      id: `edge-${index}`,
      source: arista.source,
      target: arista.target,
      lineWidth: 1 + (Math.sqrt(arista.peso) / Math.sqrt(maxWeight)) * 5,
    },
  }))

  return [...edges, ...nodes]
}

export function GraphVisualization() {
  const { data, loading, error, execute } = useApiRequest<CronogramaResponse>(true)
  const graphElement = useRef<HTMLDivElement>(null)
  const [graphError, setGraphError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    void Promise.resolve().then(() => {
      if (active) {
        return execute(() => postCronograma())
      }
    })

    return () => {
      active = false
    }
  }, [execute])

  const invalidColoring = useMemo(
    () => (data ? coloringError(data) : null),
    [data],
  )

  useEffect(() => {
    if (!data || invalidColoring || !graphElement.current || data.nodos.length === 0) {
      return
    }

    let active = true
    let instance: Core | undefined

    void import('cytoscape')
      .then(({ default: cytoscape }) => {
        if (!active || !graphElement.current) {
          return
        }

        instance = cytoscape({
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

      })
      .catch(() => {
        if (active) {
          setGraphError('No se pudo iniciar la visualización interactiva del grafo.')
        }
      })

    return () => {
      active = false
      instance?.destroy()
    }
  }, [data, invalidColoring])

  const turnos = useMemo(
    () =>
      data
        ? [...new Set(data.nodos.map((nodo) => nodo.turno))].sort((a, b) => a - b)
        : [],
    [data],
  )

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
            {data.cantidad_nodos} materias · {data.cantidad_aristas} conflictos
          </span>
        )}
      </div>

      <div className="card-body">
        {loading && (
          <div className="state-container state-loading" role="status">
            <div className="spinner" />
            <p>Generando el grafo de conflictos...</p>
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
              onClick={() => void execute(() => postCronograma())}
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

        {!loading && !error && !invalidColoring && !graphError && data && data.nodos.length > 0 && (
          <>
            <div
              className="graph-canvas"
              ref={graphElement}
              role="img"
              aria-label={`Grafo con ${data.cantidad_nodos} materias y ${data.cantidad_aristas} conflictos`}
            />
            <div className="graph-legend" aria-label="Leyenda de turnos">
              <strong>Turnos</strong>
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
