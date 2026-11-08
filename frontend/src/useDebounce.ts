import { useEffect, useState } from 'react'

/**
 * Hook para demorar la propagación de un valor hasta que transcurra
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

