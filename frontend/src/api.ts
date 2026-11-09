import type {
  ConflictosResponse,
  CronogramaRequest,
  CronogramaResponse,
  HealthResponse,
  Materia,
  MetricasResponse,
} from './types'

const rawApiUrl = (
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
).trim()

const normalizedApiUrl =
  rawApiUrl.startsWith('http://') || rawApiUrl.startsWith('https://')
    ? rawApiUrl
    : `https://${rawApiUrl}`

export const API_BASE_URL = normalizedApiUrl.replace(/\/+$/, '')

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
    if (typeof body === 'object' && body !== null && 'detail' in body) {
      if (typeof body.detail === 'string') {
        return body.detail
      }

      if (Array.isArray(body.detail)) {
        return body.detail
          .map((detail) => {
            if (
              typeof detail === 'object' &&
              detail !== null &&
              'msg' in detail &&
              typeof detail.msg === 'string'
            ) {
              return detail.msg
            }
            return JSON.stringify(detail) ?? String(detail)
          })
          .join(', ')
      }
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
  } catch (error: unknown) {
    if (init?.signal?.aborted) {
      throw error
    }
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
  } catch (error: unknown) {
    if (init?.signal?.aborted) {
      throw error
    }
    throw new ApiError('El servidor devolvió una respuesta JSON inválida.', response.status)
  }
}

export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>('/health', { signal })
}

export function getMaterias(signal?: AbortSignal): Promise<Materia[]> {
  return request<Materia[]>('/materias', { signal })
}

export function getConflictos(
  umbral = 15,
  signal?: AbortSignal,
): Promise<ConflictosResponse> {
  const query = new URLSearchParams({ umbral: String(umbral) })
  return request<ConflictosResponse>(`/conflictos?${query}`, { signal })
}

export function postCronograma(
  body: CronogramaRequest = {},
  signal?: AbortSignal,
): Promise<CronogramaResponse> {
  return request<CronogramaResponse>('/cronograma', {
    method: 'POST',
    body: JSON.stringify(body),
    signal,
  })
}

export function getMetricas(signal?: AbortSignal): Promise<MetricasResponse> {
  return request<MetricasResponse>('/metricas', { signal })
}
