import { useEffect, useState } from 'react'
import './App.css'

interface HealthResponse {
  status: string
  mensaje: string
}

export function App() {
  const [data, setData] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [latency, setLatency] = useState<number | null>(null)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
  const endpoint = `${apiUrl}/health`

  const checkHealth = async () => {
    setLoading(true)
    setError(null)
    const startTime = performance.now()

    try {
      const response = await fetch(endpoint, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      })

      const endTime = performance.now()
      setLatency(Math.round(endTime - startTime))
      setLastChecked(new Date())

      if (!response.ok) {
        throw new Error(`El servidor respondió con código HTTP ${response.status}`)
      }

      const json: HealthResponse = await response.json()
      setData(json)
    } catch (err: unknown) {
      const endTime = performance.now()
      setLatency(Math.round(endTime - startTime))
      setLastChecked(new Date())
      setData(null)

      if (err instanceof Error) {
        setError(err.message)
      } else {
        setError('No se pudo conectar con el backend. Verifique que esté corriendo.')
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
  }, [])

  return (
    <div className="app-container">
      {/* Background glow effects */}
      <div className="glow-sphere sphere-1" />
      <div className="glow-sphere sphere-2" />

      <header className="app-header">
        <div className="project-badge">
          <span className="badge-tag">TP Aviani &amp; Canal</span>
          <span className="badge-divider">•</span>
          <span className="badge-sub">Planificación de Mesas de Examen</span>
        </div>
        <h1 className="main-title">Panel de Control &amp; Estado</h1>
        <p className="main-subtitle">
          Verificación de comunicación en tiempo real entre el Frontend (Vite + React) y el Backend (FastAPI).
        </p>
      </header>

      <main className="main-content">
        <div className="card glass-card">
          <div className="card-header">
            <div className="status-indicator-wrapper">
              <span
                className={`status-dot ${
                  loading ? 'dot-loading' : error ? 'dot-error' : 'dot-success'
                }`}
              />
              <span className="status-title">
                {loading
                  ? 'Verificando conexión...'
                  : error
                  ? 'Servidor no disponible'
                  : 'Backend Conectado'}
              </span>
            </div>
            <span
              className={`status-pill ${
                loading ? 'pill-loading' : error ? 'pill-error' : 'pill-success'
              }`}
            >
              {loading ? 'CONECTANDO' : error ? 'OFFLINE' : 'ONLINE'}
            </span>
          </div>

          <div className="card-body">
            <div className="info-row">
              <span className="info-label">Endpoint consultado:</span>
              <code className="info-code" id="endpoint-url">{endpoint}</code>
            </div>

            {loading && (
              <div className="state-container state-loading" id="health-loading">
                <div className="spinner" />
                <p>Consultando estado del servidor FastAPI...</p>
              </div>
            )}

            {!loading && error && (
              <div className="state-container state-error" id="health-error">
                <div className="state-icon error-icon">✕</div>
                <div className="state-details">
                  <h3>Error de conexión</h3>
                  <p>{error}</p>
                  <small>
                    Asegurate de que el backend esté ejecutándose en <code>{apiUrl}</code> y que CORS esté activo.
                  </small>
                </div>
              </div>
            )}

            {!loading && data && (
              <div className="state-container state-success" id="health-success">
                <div className="state-icon success-icon">✓</div>
                <div className="state-details">
                  <h3>Respuesta del Backend</h3>
                  <div className="response-grid">
                    <div className="response-item">
                      <span className="item-label">Estado</span>
                      <strong className="item-value success-text">{data.status}</strong>
                    </div>
                    <div className="response-item">
                      <span className="item-label">Mensaje</span>
                      <strong className="item-value">"{data.mensaje}"</strong>
                    </div>
                    {latency !== null && (
                      <div className="response-item">
                        <span className="item-label">Latencia</span>
                        <strong className="item-value">{latency} ms</strong>
                      </div>
                    )}
                    {lastChecked && (
                      <div className="response-item">
                        <span className="item-label">Último chequeo</span>
                        <strong className="item-value">
                          {lastChecked.toLocaleTimeString()}
                        </strong>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="card-footer">
            <button
              id="btn-reintentar"
              type="button"
              className="btn btn-primary"
              onClick={checkHealth}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="btn-spinner" />
                  Verificando...
                </>
              ) : (
                <>
                  <svg
                    className="btn-icon"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                  </svg>
                  Reintentar conexión
                </>
              )}
            </button>
          </div>
        </div>

        {/* Tech Stack Cards */}
        <div className="stack-grid">
          <div className="stack-card">
            <span className="stack-role">Backend</span>
            <h4>FastAPI 0.111+</h4>
            <p>API REST para grafos y cálculo de cronogramas de examen.</p>
          </div>
          <div className="stack-card">
            <span className="stack-role">Frontend</span>
            <h4>React 19 + TypeScript</h4>
            <p>SPA moderna montada con Vite para máxima velocidad de desarrollo.</p>
          </div>
          <div className="stack-card">
            <span className="stack-role">CORS</span>
            <h4>Configurado</h4>
            <p>Permite origen seguro <code>http://localhost:5173</code>.</p>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>Trabajo Práctico — Planificación de Mesas de Examen con Coloreo de Grafos</p>
      </footer>
    </div>
  )
}

export default App
