import type {
  ConflictosResponse,
  CronogramaRequest,
  CronogramaResponse,
  HealthResponse,
  Materia,
  MetricasResponse,
} from './types'

export const API_BASE_URL = (
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
).replace(/\/+$/, '')

export class ApiError extends Error {
  public readonly status: number

  constructor(
    message: string,
    status: number,
  ) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function getErrorMessage(response: Response): Promise<string> {
  const responseText = await response.text()

  if (!responseText) {
    return `El servidor respondió con código HTTP ${response.status}`
  }

  try {
    const body: unknown = JSON.parse(responseText)
    if (
      typeof body === 'object' &&
      body !== null &&
      'detail' in body &&
      typeof body.detail === 'string'
    ) {
      return body.detail
    }
  } catch {
    return responseText
  }

  return responseText
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers })
  } catch {
    throw new ApiError(
      'No se pudo conectar con el backend. Verifique que esté corriendo.',
      0,
    )
  }

  if (!response.ok) {
    throw new ApiError(await getErrorMessage(response), response.status)
  }

  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError('El servidor devolvió una respuesta JSON inválida.', response.status)
  }
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function getMaterias(): Promise<Materia[]> {
  return request<Materia[]>('/materias')
}

export function getConflictos(umbral = 15): Promise<ConflictosResponse> {
  const query = new URLSearchParams({ umbral: String(umbral) })
  return request<ConflictosResponse>(`/conflictos?${query}`)
}

export function postCronograma(
  body: CronogramaRequest = {},
): Promise<CronogramaResponse> {
  return request<CronogramaResponse>('/cronograma', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function getMetricas(): Promise<MetricasResponse> {
  return request<MetricasResponse>('/metricas')
}
