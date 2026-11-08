import { useCallback, useEffect, useRef, useState } from 'react'

// Hook simple de debounce para que el slider o el buscador no disparen
// pedidos a la API en cada micro-cambio.
export function useDebounce<T>(value: T, delayMs = 350): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value)

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value)
    }, delayMs)

    return () => {
      clearTimeout(handler)
    }
  }, [value, delayMs])

  return debouncedValue
}

export interface DebouncedCallback<Args extends unknown[]> {
  run: (...args: Args) => void
  cancel: () => void
  flush: (...args: Args) => void
}

// Variante que envuelve una función con debounce y permite cancelar o ejecutar ya
export function useDebouncedCallback<Args extends unknown[]>(
  callback: (...args: Args) => void,
  delayMs = 350,
): DebouncedCallback<Args> {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const callbackRef = useRef<(...args: Args) => void>(callback)

  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  const cancel = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
  }, [])

  const run = useCallback(
    (...args: Args) => {
      cancel()
      timerRef.current = setTimeout(() => {
        callbackRef.current(...args)
        timerRef.current = null
      }, delayMs)
    },
    [cancel, delayMs],
  )

  const flush = useCallback(
    (...args: Args) => {
      cancel()
      callbackRef.current(...args)
    },
    [cancel],
  )

  useEffect(() => {
    return cancel
  }, [cancel])

  return { run, cancel, flush }
}
