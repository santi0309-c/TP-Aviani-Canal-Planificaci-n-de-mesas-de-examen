import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Hook para demorar la propagación de un valor reactivo hasta que transcurra
 * el intervalo especificado sin nuevas modificaciones (Debounce).
 *
 * @param value Valor reactivo a demorar.
 * @param delayMs Tiempo de espera en milisegundos (por defecto 350ms).
 * @returns El valor estabilizado tras el debounce.
 */
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

/**
 * Hook para ejecutar un callback con debounce y control explícito
 * de cancelación y ejecución inmediata (flush).
 *
 * @param callback Función a ejecutar.
 * @param delayMs Tiempo de espera en milisegundos (por defecto 350ms).
 * @returns Objeto con métodos run (demorado), cancel (cancelar timer) y flush (inmediato).
 */
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
