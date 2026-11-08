import { useCallback, useState } from 'react'
import { API_BASE_URL, getHealth, postCronograma } from './api'
import { CronogramaGrid } from './CronogramaGrid'
import { GraphVisualization } from './GraphVisualization'
import { MetricsPanel } from './MetricsPanel'
import type { CronogramaResponse, HealthResponse } from './types'
import { useApiRequest } from './useApiRequest'
import './App.css'

export function App() {
  const { data: healthData, loading: healthLoading, error: healthError, status: healthStatus, execute: executeHealth } =
    useApiRequest<HealthResponse>(getHealth)
  const [latency, setLatency] = useState<number | null>(null)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)
  const [showHealthDetails, setShowHealthDetails] = useState<boolean>(false)
  const [showGuide, setShowGuide] = useState<boolean>(true)

  const [umbral, setUmbral] = useState<number>(15)

  const loadCronograma = useCallback(
    (signal: AbortSignal) => postCronograma({ umbral }, signal),
    [umbral],
  )

  const {
    data: cronograma,
    loading: cronogramaLoading,
    error: cronogramaError,
    status: cronogramaStatus,
    execute: executeCronograma,
  } = useApiRequest<CronogramaResponse>(loadCronograma)

  const handleCronogramaRetry = useCallback(() => {
    void executeCronograma(loadCronograma).catch(() => {})
  }, [executeCronograma, loadCronograma])

  const endpoint = `${API_BASE_URL}/health`

  const checkHealth = useCallback(async () => {
    const startTime = performance.now()

    try {
      await executeHealth(getHealth)
    } catch {
      // The hook exposes request failures in the connection status indicator.
    } finally {
      setLatency(Math.round(performance.now() - startTime))
      setLastChecked(new Date())
    }
  }, [executeHealth])

  return (
    <div className="app-container">
      {/* Ambient background glow effects */}
      <div className="glow-sphere sphere-1" />
      <div className="glow-sphere sphere-2" />

      {/* Header y Barra de Estado Superior */}
      <header className="app-header">
        <div className="header-top-bar">
          <div className="project-badge">
            <span className="badge-tag">TP Aviani &amp; Canal</span>
            <span className="badge-divider">•</span>
            <span className="badge-sub">Planificación de Mesas de Examen</span>
          </div>

          {/* Indicador de Conexión Compacto para la Demo */}
          <div className="server-status-pill-group">
            <div
              className={`server-status-pill ${
                healthLoading ? 'server-loading' : healthError ? 'server-error' : 'server-online'
              }`}
              title={`Endpoint: ${endpoint}`}
            >
              <span
                className={`status-dot ${
                  healthLoading ? 'dot-loading' : healthError ? 'dot-error' : 'dot-success'
                }`}
              />
              <span className="server-status-label">
                {healthLoading
                  ? 'Verificando API...'
                  : healthError
                  ? 'Backend Offline'
                  : 'FastAPI Conectado'}
              </span>
              {latency !== null && !healthError && !healthLoading && (
                <span className="server-latency">{latency}ms</span>
              )}
            </div>

            <button
              type="button"
              className="btn-status-toggle"
              onClick={() => setShowHealthDetails(!showHealthDetails)}
              aria-label="Ver detalles de conexión con el backend"
              title="Ver detalles del servidor"
            >
              {showHealthDetails ? 'Cerrar info' : 'Detalles'}
            </button>
          </div>
        </div>

        <h1 className="main-title">Planificación de Mesas de Examen</h1>
        <p className="main-subtitle">
          Resolución del problema de asignación de turnos universitarios combinando <strong>Machine Learning</strong> (predicción de estudiantes afectados) y <strong>Coloreo de Grafos</strong> (algoritmo greedy Welsh-Powell).
        </p>

        {/* Panel Desplegable de Detalles de Conexión */}
        {showHealthDetails && (
          <div className="server-details-drawer glass-card" role="region" aria-label="Detalles de conexión">
            <div className="details-header">
              <h4>Estado de Conexión de la API</h4>
              <button
                type="button"
                className="btn-close-drawer"
                onClick={() => setShowHealthDetails(false)}
              >
                ✕
              </button>
            </div>
            <div className="details-body">
              <div className="info-row">
                <span className="info-label">URL del Endpoint:</span>
                <code className="info-code">{endpoint}</code>
              </div>
              {healthData && (
                <div className="response-grid">
                  <div className="response-item">
                    <span className="item-label">Respuesta</span>
                    <strong className="item-value success-text">{healthData.status}</strong>
                  </div>
                  <div className="response-item">
                    <span className="item-label">Mensaje</span>
                    <strong className="item-value">"{healthData.mensaje}"</strong>
                  </div>
                  {lastChecked && (
                    <div className="response-item">
                      <span className="item-label">Último chequeo</span>
                      <strong className="item-value">{lastChecked.toLocaleTimeString()}</strong>
                    </div>
                  )}
                </div>
              )}
              {healthError && (
                <div className="state-container state-error">
                  <div className="state-details">
                    <p>{healthError}</p>
                    {healthStatus !== null && <small>Código HTTP: {healthStatus}</small>}
                  </div>
                </div>
              )}
            </div>
            <div className="details-footer">
              <button
                type="button"
                className="btn btn-primary"
                onClick={checkHealth}
                disabled={healthLoading}
              >
                {healthLoading ? 'Reverificando...' : 'Reintentar chequeo'}
              </button>
            </div>
          </div>
        )}
      </header>

      <main className="main-content">
        {/* Banner Pedagógico si el Backend está Desconectado */}
        {healthError && (
          <div className="backend-offline-banner glass-card" role="alert">
            <div className="banner-alert-icon">⚠️</div>
            <div className="banner-alert-content">
              <h3>Servidor Backend no disponible en <code>{API_BASE_URL}</code></h3>
              <p>
                Para levantar la API FastAPI y permitir la generación dinámica del cronograma, ejecutá en tu terminal:
              </p>
              <pre className="command-snippet">
                <code>cd backend && uvicorn app.main:app --reload --port 8000</code>
              </pre>
            </div>
            <button
              type="button"
              className="btn btn-primary banner-retry-btn"
              onClick={checkHealth}
              disabled={healthLoading}
            >
              Reintentar conexión
            </button>
          </div>
        )}

        {/* Guía Visual Interactiva para la Defensa Académica */}
        <div className="guide-card glass-card">
          <button
            type="button"
            className="guide-toggle-btn"
            onClick={() => setShowGuide(!showGuide)}
            aria-expanded={showGuide}
          >
            <div className="guide-btn-title">
              <span className="guide-icon">💡</span>
              <span>Guía para la Defensa: ¿Cómo funciona este sistema?</span>
            </div>
            <span className="guide-chevron">{showGuide ? '▲ Ocultar guía' : '▼ Ver guía didáctica'}</span>
          </button>

          {showGuide && (
            <div className="guide-content">
              <div className="guide-grid">
                <div className="guide-step">
                  <div className="step-badge">1. Machine Learning</div>
                  <h4>Predicción de Conflictos</h4>
                  <p>
                    Un modelo de regresión infiere la cantidad estimada de estudiantes que rendirían cada par de materias a partir de datos históricos.
                  </p>
                </div>
                <div className="guide-step">
                  <div className="step-badge">2. Umbral de Corte (τ)</div>
                  <h4>Compromiso (Trade-off)</h4>
                  <p>
                    Se crea una arista restrictiva solo si el conflicto predicho supera el umbral $\tau$. Menor $\tau$ = más turnos y cero colisiones; mayor $\tau$ = calendario compacto.
                  </p>
                </div>
                <div className="guide-step">
                  <div className="step-badge">3. Coloreo de Grafos</div>
                  <h4>Algoritmo Welsh-Powell</h4>
                  <p>
                    Cada nodo es una materia y su color representa el turno de examen. Materias unidas por aristas <strong>nunca</strong> comparten color ni turno.
                  </p>
                </div>
                <div className="guide-step">
                  <div className="step-badge">4. Planificación Final</div>
                  <h4>Cronograma Óptimo</h4>
                  <p>
                    El número cromático resultante determina la cantidad mínima de turnos de examen necesarios sin superposición de fechas.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 1. Grafo de Conflictos con Selector de Umbral y Presets */}
        <GraphVisualization
          umbral={umbral}
          onUmbralChange={setUmbral}
          cronograma={cronograma}
          loading={cronogramaLoading}
          error={cronogramaError}
          status={cronogramaStatus}
          onRetry={handleCronogramaRetry}
        />

        {/* 2. Panel Central de Métricas y Evaluación de ML */}
        <MetricsPanel
          cronograma={cronograma}
          isUpdating={cronogramaLoading && Boolean(cronograma)}
        />

        {/* 3. Distribución de Turnos de Examen en Grilla */}
        <CronogramaGrid
          cronograma={cronograma}
          isUpdating={cronogramaLoading && Boolean(cronograma)}
          loading={cronogramaLoading && !cronograma}
          error={cronogramaError}
          onRetry={handleCronogramaRetry}
        />

        {/* Stack Tecnológico del Proyecto */}
        <div className="stack-grid">
          <div className="stack-card">
            <span className="stack-role">Backend</span>
            <h4>FastAPI 0.111+</h4>
            <p>API REST de alto rendimiento con Pydantic y documentación interactiva OpenAPI.</p>
          </div>
          <div className="stack-card">
            <span className="stack-role">Machine Learning</span>
            <h4>Scikit-Learn</h4>
            <p>Modelos de regresión y clasificación entrenados para inferir conflictos de cursada.</p>
          </div>
          <div className="stack-card">
            <span className="stack-role">Algoritmos</span>
            <h4>Coloreo de Grafos</h4>
            <p>Implementación greedy con heurística de ordenamiento por grado Welsh-Powell.</p>
          </div>
          <div className="stack-card">
            <span className="stack-role">Frontend</span>
            <h4>React 19 + TypeScript + Cytoscape</h4>
            <p>SPA moderna con visualización interactiva y reactividad dual a 60 FPS.</p>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Trabajo Práctico — Planificación de Mesas de Examen mediante Grafos y Machine Learning</p>
        <p className="footer-sub">Ingeniería en Sistemas de Información · Valentino Aviani &amp; Santiago Cañal</p>
      </footer>
    </div>
  )
}

export default App
