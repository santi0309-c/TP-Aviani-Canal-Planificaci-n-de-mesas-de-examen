import { useId, useMemo, useState } from 'react'
import { turnoColor } from './colors'
import type { CronogramaResponse, NodoCronograma } from './types'

export interface CronogramaGridProps {
  cronograma: CronogramaResponse | null
  isUpdating?: boolean
  loading?: boolean
  error?: string | null
  onRetry?: () => void
}

interface TurnoGroup {
  turno: number
  label: string
  color: string
  materias: NodoCronograma[]
  cantidad: number
}

export function CronogramaGrid({
  cronograma,
  isUpdating = false,
  loading = false,
  error = null,
  onRetry,
}: CronogramaGridProps) {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const searchInputId = useId()

  const turnosAgrupados = useMemo((): TurnoGroup[] => {
    if (!cronograma || !cronograma.nodos.length) {
      return []
    }

    const mapa = new Map<number, NodoCronograma[]>()
    for (const nodo of cronograma.nodos) {
      const lista = mapa.get(nodo.turno) ?? []
      lista.push(nodo)
      mapa.set(nodo.turno, lista)
    }

    const turnosOrdenados = Array.from(mapa.keys()).sort((a, b) => a - b)

    return turnosOrdenados.map((turno) => {
      const materias = mapa.get(turno) ?? []
      materias.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
      return {
        turno,
        label: `Turno ${turno + 1}`,
        color: turnoColor(turno),
        materias,
        cantidad: materias.length,
      }
    })
  }, [cronograma])

  const normalizedSearch = searchTerm.trim().toLowerCase()

  const promedioMateriasPorTurno = useMemo(() => {
    if (!cronograma || cronograma.cantidad_turnos === 0) {
      return '0'
    }
    return (cronograma.cantidad_nodos / cronograma.cantidad_turnos).toFixed(1)
  }, [cronograma])

  const totalMateriasEncontradas = useMemo(() => {
    if (!normalizedSearch || !cronograma) {
      return null
    }
    return cronograma.nodos.filter(
      (nodo) =>
        nodo.nombre.toLowerCase().includes(normalizedSearch) ||
        nodo.id.toLowerCase().includes(normalizedSearch),
    ).length
  }, [cronograma, normalizedSearch])

  return (
    <section
      className="card glass-card cronograma-card"
      aria-labelledby="cronograma-title"
    >
      <div className="card-header">
        <div className="cronograma-title-group">
          <div className="cronograma-title-row">
            <h2 className="status-title" id="cronograma-title">
              Distribución por turnos de examen
            </h2>
            {isUpdating && (
              <span className="cronograma-updating-pill" aria-live="polite">
                <span className="btn-spinner" />
                Actualizando grilla...
              </span>
            )}
          </div>
          <p className="cronograma-subtitle">
            Materias asignadas a cada turno para evitar superposiciones de examen.
            El color de cada turno coincide con el coloreo del grafo.
          </p>
        </div>

        {cronograma && (
          <div className="cronograma-summary-badges">
            <span className="cronograma-summary-badge" title="Cantidad total de turnos">
              <strong>{cronograma.cantidad_turnos}</strong> turnos
            </span>
            <span className="cronograma-summary-badge" title="Cantidad total de materias planificadas">
              <strong>{cronograma.cantidad_nodos}</strong> materias
            </span>
            <span className="cronograma-summary-badge" title="Promedio de materias por cada turno">
              <strong>~{promedioMateriasPorTurno}</strong> mat/turno
            </span>
          </div>
        )}
      </div>

      <div className="card-body">
        {/* Barra de herramientas: Búsqueda y filtrado rápido */}
        {cronograma && cronograma.nodos.length > 0 && (
          <div className="cronograma-toolbar">
            <div className="cronograma-search-wrapper">
              <label htmlFor={searchInputId} className="visually-hidden">
                Buscar materia por nombre o código
              </label>
              <input
                id={searchInputId}
                type="search"
                className="cronograma-search-input"
                placeholder="Buscar materia (ej. Análisis, M01)..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              {searchTerm && (
                <button
                  type="button"
                  className="cronograma-search-clear"
                  onClick={() => setSearchTerm('')}
                  aria-label="Limpiar búsqueda"
                >
                  ✕
                </button>
              )}
            </div>

            {normalizedSearch && totalMateriasEncontradas !== null && (
              <span className="cronograma-search-results" aria-live="polite">
                {totalMateriasEncontradas === 0
                  ? 'No se encontraron materias con ese término.'
                  : `${totalMateriasEncontradas} ${
                      totalMateriasEncontradas === 1 ? 'materia encontrada' : 'materias encontradas'
                    }`}
              </span>
            )}
          </div>
        )}

        {/* Estados de Carga y Error */}
        {loading && !cronograma && (
          <div className="state-container state-loading" role="status">
            <div className="spinner" />
            <p>Generando la distribución de turnos de examen...</p>
          </div>
        )}

        {!loading && error && (
          <div className="state-container state-error" role="alert">
            <div className="state-icon error-icon">✕</div>
            <div className="state-details">
              <h3>No se pudo cargar el cronograma</h3>
              <p>{error}</p>
            </div>
            {onRetry && (
              <button className="btn btn-primary" type="button" onClick={onRetry}>
                Reintentar
              </button>
            )}
          </div>
        )}

        {!loading && !error && cronograma && cronograma.nodos.length === 0 && (
          <div className="state-container">
            <p>No hay materias asignadas en el cronograma actual.</p>
          </div>
        )}

        {/* Grilla de Columnas por Turno */}
        {cronograma && turnosAgrupados.length > 0 && (
          <div
            className="cronograma-columns-container"
            role="region"
            aria-label="Grilla interactiva de turnos de examen"
          >
            <div className="cronograma-columns-grid">
              {turnosAgrupados.map((group) => {
                const materiasCoincidentes = normalizedSearch
                  ? group.materias.filter(
                      (m) =>
                        m.nombre.toLowerCase().includes(normalizedSearch) ||
                        m.id.toLowerCase().includes(normalizedSearch),
                    )
                  : group.materias

                const hasMatches = materiasCoincidentes.length > 0
                const isFilteredOut = normalizedSearch.length > 0 && !hasMatches

                return (
                  <div
                    key={group.turno}
                    className={`turno-column ${isFilteredOut ? 'turno-dimmed' : ''}`}
                    style={
                      {
                        '--turno-theme-color': group.color,
                      } as React.CSSProperties
                    }
                  >
                    <div className="turno-column-header">
                      <div className="turno-header-title">
                        <span
                          className="turno-indicator-dot"
                          style={{ backgroundColor: group.color }}
                          aria-hidden="true"
                        />
                        <h3>{group.label}</h3>
                      </div>
                      <span
                        className="turno-count-badge"
                        style={{
                          borderColor: group.color,
                        }}
                      >
                        {group.cantidad} {group.cantidad === 1 ? 'materia' : 'materias'}
                      </span>
                    </div>

                    <ul
                      className="turno-materias-list"
                      role="list"
                      aria-label={`Materias del ${group.label}`}
                    >
                      {group.materias.map((materia) => {
                        const isMatch =
                          normalizedSearch &&
                          (materia.nombre.toLowerCase().includes(normalizedSearch) ||
                            materia.id.toLowerCase().includes(normalizedSearch))

                        return (
                          <li
                            key={materia.id}
                            className={`materia-item ${isMatch ? 'materia-item-matched' : ''}`}
                            role="listitem"
                          >
                            <div className="materia-header-row">
                              <span
                                className="materia-code-pill"
                                style={{
                                  backgroundColor: `${group.color}22`,
                                  borderColor: `${group.color}66`,
                                  color: '#ffffff',
                                }}
                              >
                                {materia.id}
                              </span>
                              {materia.anio && (
                                <span className="materia-anio-tag">
                                  {materia.anio}° año
                                </span>
                              )}
                            </div>
                            <span className="materia-nombre-text" title={materia.nombre}>
                              {materia.nombre}
                            </span>
                          </li>
                        )
                      })}
                    </ul>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

