import { useCallback, useMemo, useState } from 'react'
import { getMetricas } from './api'
import type { CronogramaResponse, MetricasResponse } from './types'
import { useApiRequest } from './useApiRequest'

export interface MetricsPanelProps {
  cronograma: CronogramaResponse | null
  isUpdating?: boolean
}

export function MetricsPanel({ cronograma, isUpdating = false }: MetricsPanelProps) {
  const [activeTab, setActiveTab] = useState<'cronograma' | 'modelo'>('cronograma')

  const loadMetricas = useCallback(
    (signal: AbortSignal) => getMetricas(signal),
    [],
  )

  const {
    data: metricas,
    loading: loadingMetricas,
    error: errorMetricas,
    execute: executeMetricas,
  } = useApiRequest<MetricasResponse>(loadMetricas)

  const handleRetryMetricas = useCallback(() => {
    void executeMetricas(loadMetricas).catch(() => {})
  }, [executeMetricas, loadMetricas])

  // Cálculos dinámicos del cronograma reactivos al umbral τ
  const kpis = useMemo(() => {
    if (!cronograma) {
      return null
    }

    const totalNodos = cronograma.cantidad_nodos
    const totalParesPosibles = (totalNodos * (totalNodos - 1)) / 2
    const totalAristas = cronograma.cantidad_aristas
    const cantidadTurnos = cronograma.cantidad_turnos

    const estudiantesProtegidos = cronograma.aristas.reduce(
      (sum, edge) => sum + edge.peso,
      0,
    )

    const pesoPromedioConflicto =
      totalAristas > 0 ? estudiantesProtegidos / totalAristas : 0

    const densidadGrafo =
      totalParesPosibles > 0 ? (totalAristas / totalParesPosibles) * 100 : 0

    const promedioMateriasTurno =
      cantidadTurnos > 0 ? totalNodos / cantidadTurnos : 0

    const maxConflicto =
      cronograma.aristas.length > 0
        ? Math.max(...cronograma.aristas.map((a) => a.peso))
        : 0

    return {
      totalNodos,
      totalAristas,
      cantidadTurnos,
      totalParesPosibles,
      estudiantesProtegidos: Math.round(estudiantesProtegidos),
      pesoPromedioConflicto: pesoPromedioConflicto.toFixed(1),
      densidadGrafo: densidadGrafo.toFixed(1),
      promedioMateriasTurno: promedioMateriasTurno.toFixed(1),
      maxConflicto: maxConflicto.toFixed(1),
      umbral: cronograma.umbral,
    }
  }, [cronograma])

  return (
    <section
      className="card glass-card metrics-panel-card"
      aria-labelledby="metrics-panel-title"
    >
      <div className="card-header">
        <div className="metrics-header-titles">
          <div className="metrics-title-row">
            <h2 className="status-title" id="metrics-panel-title">
              Panel de Métricas y Rendimiento
            </h2>
            {isUpdating && (
              <span className="metrics-updating-indicator" aria-live="polite">
                <span className="btn-spinner" />
                Actualizando KPIs...
              </span>
            )}
          </div>
          <p className="metrics-subtitle">
            Indicadores clave del cronograma calculados en tiempo real y métricas de evaluación del modelo de Machine Learning.
          </p>
        </div>

        {/* Selector de pestañas */}
        <div className="metrics-nav-tabs" role="tablist" aria-label="Secciones del panel de métricas">
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'cronograma'}
            className={`metrics-tab-btn ${activeTab === 'cronograma' ? 'metrics-tab-active' : ''}`}
            onClick={() => setActiveTab('cronograma')}
          >
            <span>Impacto del Umbral</span>
            {kpis && <span className="tab-pill">τ = {kpis.umbral}</span>}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === 'modelo'}
            className={`metrics-tab-btn ${activeTab === 'modelo' ? 'metrics-tab-active' : ''}`}
            onClick={() => setActiveTab('modelo')}
          >
            <span>Métricas del Modelo (ML)</span>
            {metricas && <span className="tab-pill tab-pill-success">R² {Math.round(metricas.regresion.r2 * 100)}%</span>}
          </button>
        </div>
      </div>

      <div className="card-body">
        {/* PESTAÑA 1: KPIS DINÁMICOS DEL CRONOGRAMA */}
        {activeTab === 'cronograma' && (
          <div className="metrics-content-fade" role="tabpanel">
            {!kpis && (
              <div className="state-container state-loading">
                <div className="spinner" />
                <p>Calculando estadísticas del cronograma...</p>
              </div>
            )}

            {kpis && (
              <div className="kpi-grid">
                {/* Turnos */}
                <div className="kpi-card kpi-card-highlight">
                  <span className="kpi-label">Turnos Necesarios</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number text-accent-violet">
                      {kpis.cantidadTurnos}
                    </span>
                    <span className="kpi-unit">turnos</span>
                  </div>
                  <span className="kpi-help">
                    Turnos asignados mediante el algoritmo Welsh-Powell para evitar colisiones.
                  </span>
                </div>

                {/* Conflictos Activos */}
                <div className="kpi-card">
                  <span className="kpi-label">Conflictos Activos (Aristas)</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number text-accent-cyan">
                      {kpis.totalAristas}
                    </span>
                    <span className="kpi-unit">/ {kpis.totalParesPosibles} pares</span>
                  </div>
                  <div className="kpi-progress-bar-bg">
                    <div
                      className="kpi-progress-bar-fill"
                      style={{ width: `${Math.min(100, Math.max(5, parseFloat(kpis.densidadGrafo)))}%` }}
                    />
                  </div>
                  <span className="kpi-help">
                    Densidad del grafo: {kpis.densidadGrafo}% de los pares con conflicto ≥ {kpis.umbral}.
                  </span>
                </div>

                {/* Materias Planificadas */}
                <div className="kpi-card">
                  <span className="kpi-label">Materias Planificadas</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number">{kpis.totalNodos}</span>
                    <span className="kpi-unit">materias</span>
                  </div>
                  <span className="kpi-help">
                    Promedio de {kpis.promedioMateriasTurno} materias distribuidas por turno.
                  </span>
                </div>

                {/* Estudiantes Protegidos */}
                <div className="kpi-card">
                  <span className="kpi-label">Estudiantes Protegidos</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number text-accent-emerald">
                      {kpis.estudiantesProtegidos}
                    </span>
                    <span className="kpi-unit">estudiantes</span>
                  </div>
                  <span className="kpi-help">
                    Suma ponderada de conflictos evitados al separar turnos.
                  </span>
                </div>

                {/* Magnitud Promedio de Conflicto */}
                <div className="kpi-card">
                  <span className="kpi-label">Conflicto Promedio</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number">{kpis.pesoPromedioConflicto}</span>
                    <span className="kpi-unit">estudiantes/par</span>
                  </div>
                  <span className="kpi-help">
                    Conflicto máximo registrado: {kpis.maxConflicto} estudiantes.
                  </span>
                </div>

                {/* Umbral de Corte y Riesgo Residual */}
                <div className="kpi-card">
                  <span className="kpi-label">Riesgo Residual Tolerado</span>
                  <div className="kpi-value-row">
                    <span className="kpi-number text-accent-amber">
                      &lt; {kpis.umbral}
                    </span>
                    <span className="kpi-unit">alumnos</span>
                  </div>
                  <span className="kpi-help">
                    Pares con cruces menores a {kpis.umbral} pueden coincidir en el mismo turno.
                  </span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* PESTAÑA 2: MÉTRICAS DEL MODELO DE MACHINE LEARNING */}
        {activeTab === 'modelo' && (
          <div className="metrics-content-fade" role="tabpanel">
            {loadingMetricas && (
              <div className="state-container state-loading" role="status">
                <div className="spinner" />
                <p>Consultando métricas de evaluación del modelo en el backend...</p>
              </div>
            )}

            {!loadingMetricas && errorMetricas && (
              <div className="state-container state-error" role="alert">
                <div className="state-icon error-icon">✕</div>
                <div className="state-details">
                  <h3>No se pudieron cargar las métricas del modelo</h3>
                  <p>{errorMetricas}</p>
                </div>
                <button
                  className="btn btn-primary"
                  type="button"
                  onClick={handleRetryMetricas}
                >
                  Reintentar
                </button>
              </div>
            )}

            {!loadingMetricas && !errorMetricas && metricas && (
              <div className="model-metrics-container">
                {/* Metadatos del Modelo */}
                <div className="model-banner">
                  <div className="model-banner-info">
                    <span className="model-banner-tag">Modelo Seleccionado</span>
                    <h3 className="model-banner-title">{metricas.metadatos.modelo}</h3>
                  </div>
                  <div className="model-banner-specs">
                    <span>
                      Muestras de Test: <strong>{metricas.metadatos.n_test}</strong>
                    </span>
                    <span className="spec-separator">•</span>
                    <span>
                      Semilla: <code>seed={metricas.metadatos.semilla}</code>
                    </span>
                  </div>
                </div>

                {/* Grid de Regresión y Clasificación */}
                <div className="metrics-duo-grid">
                  {/* Regresión */}
                  <div className="metric-box">
                    <div className="metric-box-header">
                      <h4>Métricas de Regresión (Cantidad de Afectados)</h4>
                      <span className="metric-badge-type">Variable Continua</span>
                    </div>

                    <div className="metric-box-stats">
                      <div className="stat-pill">
                        <span className="stat-name">R² Score</span>
                        <strong className="stat-value text-accent-emerald">
                          {(metricas.regresion.r2 * 100).toFixed(2)}%
                        </strong>
                        <small className="stat-desc">Varianza explicada</small>
                      </div>

                      <div className="stat-pill">
                        <span className="stat-name">MAE</span>
                        <strong className="stat-value text-accent-cyan">
                          {metricas.regresion.mae.toFixed(2)}
                        </strong>
                        <small className="stat-desc">Error absoluto medio</small>
                      </div>

                      <div className="stat-pill">
                        <span className="stat-name">RMSE</span>
                        <strong className="stat-value">
                          {metricas.regresion.rmse.toFixed(2)}
                        </strong>
                        <small className="stat-desc">Raíz del error cuadrático</small>
                      </div>
                    </div>
                  </div>

                  {/* Clasificación */}
                  <div className="metric-box">
                    <div className="metric-box-header">
                      <h4>Métricas de Clasificación (Hubo Conflicto ≥ {metricas.clasificacion.umbral})</h4>
                      <span className="metric-badge-type">Decisión Binaria</span>
                    </div>

                    <div className="metric-box-stats">
                      <div className="stat-pill">
                        <span className="stat-name">Accuracy</span>
                        <strong className="stat-value text-accent-violet">
                          {(metricas.clasificacion.accuracy * 100).toFixed(1)}%
                        </strong>
                        <small className="stat-desc">Aciertos totales</small>
                      </div>

                      <div className="stat-pill">
                        <span className="stat-name">F1-Score</span>
                        <strong className="stat-value text-accent-emerald">
                          {(metricas.clasificacion.f1 * 100).toFixed(1)}%
                        </strong>
                        <small className="stat-desc">Media armónica</small>
                      </div>

                      <div className="stat-pill">
                        <span className="stat-name">Precision</span>
                        <strong className="stat-value">
                          {(metricas.clasificacion.precision * 100).toFixed(1)}%
                        </strong>
                        <small className="stat-desc">Verdaderos positivos</small>
                      </div>

                      <div className="stat-pill">
                        <span className="stat-name">Recall</span>
                        <strong className="stat-value">
                          {(metricas.clasificacion.recall * 100).toFixed(1)}%
                        </strong>
                        <small className="stat-desc">Sensibilidad</small>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Matriz de Confusión Resumida */}
                <div className="confusion-matrix-wrapper">
                  <h4 className="matrix-title">Matriz de Confusión (Test Set: {metricas.matriz_confusion.total} pares)</h4>
                  <div className="matrix-cells-grid">
                    <div className="matrix-cell cell-tp">
                      <span className="cell-label">Verdaderos Positivos</span>
                      <strong className="cell-num">{metricas.matriz_confusion.verdaderos_positivos}</strong>
                      <span className="cell-sub">Conflictos predichos y confirmados</span>
                    </div>
                    <div className="matrix-cell cell-fn">
                      <span className="cell-label">Falsos Negativos</span>
                      <strong className="cell-num">{metricas.matriz_confusion.falsos_negativos}</strong>
                      <span className="cell-sub">Conflictos no detectados</span>
                    </div>
                    <div className="matrix-cell cell-fp">
                      <span className="cell-label">Falsos Positivos</span>
                      <strong className="cell-num">{metricas.matriz_confusion.falsos_positivos}</strong>
                      <span className="cell-sub">Falsas alarmas</span>
                    </div>
                    <div className="matrix-cell cell-tn">
                      <span className="cell-label">Verdaderos Negativos</span>
                      <strong className="cell-num">{metricas.matriz_confusion.verdaderos_negativos}</strong>
                      <span className="cell-sub">Pares sin conflicto confirmados</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  )
}

