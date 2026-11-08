import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError } from './api'

interface ApiRequestState<T> {
  data: T | null
  loading: boolean
  error: string | null
  status: number | null
}

type ApiRequest<T> = (signal: AbortSignal) => Promise<T>

export function useApiRequest<T>(initialRequest?: ApiRequest<T>) {
  const [state, setState] = useState<ApiRequestState<T>>({
    data: null,
    loading: initialRequest !== undefined,
    error: null,
    status: null,
  })
  const activeController = useRef<AbortController | null>(null)

  useEffect(() => {
    return () => {
      activeController.current?.abort()
      activeController.current = null
    }
  }, [])

  const execute = useCallback(async (request: ApiRequest<T>): Promise<T> => {
    activeController.current?.abort()
    const controller = new AbortController()
    activeController.current = controller
    setState((current) => ({
      ...current,
      loading: true,
      error: null,
      status: null,
    }))

    try {
      const data = await request(controller.signal)
      if (activeController.current === controller) {
        setState({ data, loading: false, error: null, status: null })
      }
      return data
    } catch (error: unknown) {
      if (activeController.current === controller && !controller.signal.aborted) {
        setState({
          data: null,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : 'Ocurrió un error inesperado al consultar la API.',
          status: error instanceof ApiError ? error.status : null,
        })
      }
      throw error
    }
  }, [])

  useEffect(() => {
    if (!initialRequest) {
      return
    }

    let active = true
    void Promise.resolve().then(() => {
      if (active) {
        void execute(initialRequest).catch(() => {
          // The hook exposes the failure through its error and status state.
        })
      }
    })

    return () => {
      active = false
    }
  }, [execute, initialRequest])

  return { ...state, execute }
}
