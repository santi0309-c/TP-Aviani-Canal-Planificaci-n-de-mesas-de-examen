import { useCallback, useEffect, useRef, useState } from 'react'

interface ApiRequestState<T> {
  data: T | null
  loading: boolean
  error: string | null
}

export function useApiRequest<T>(initialLoading = false) {
  const [state, setState] = useState<ApiRequestState<T>>({
    data: null,
    loading: initialLoading,
    error: null,
  })
  const requestId = useRef(0)

  useEffect(() => {
    return () => {
      requestId.current += 1
    }
  }, [])

  const execute = useCallback(async (request: () => Promise<T>) => {
    const currentRequestId = ++requestId.current
    setState((current) => ({ ...current, loading: true, error: null }))

    try {
      const data = await request()
      if (requestId.current === currentRequestId) {
        setState({ data, loading: false, error: null })
      }
    } catch (error: unknown) {
      if (requestId.current === currentRequestId) {
        setState({
          data: null,
          loading: false,
          error:
            error instanceof Error
              ? error.message
              : 'Ocurrió un error inesperado al consultar la API.',
        })
      }
    }
  }, [])

  return { ...state, execute }
}
