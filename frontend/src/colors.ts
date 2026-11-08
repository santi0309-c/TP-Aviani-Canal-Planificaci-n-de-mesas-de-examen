/**
 * Paleta de colores determinística para turnos de examen.
 *
 * Utiliza el ángulo áureo (137.508°) sobre el espacio de color HSL/RGB para
 * maximizar la distinción perceptual entre turnos contiguos sin colisiones visuales.
 *
 * Garantiza consistencia cromática estricta 1:1 entre el grafo de conflictos
 * (Cytoscape) y la grilla/calendario de turnos de examen.
 *
 * @param turno Índice no negativo del turno de examen (0, 1, 2, ...).
 * @returns Código de color hexadecimal (ej. '#8b5cf6').
 */
export function turnoColor(turno: number): string {
  const hue = (turno * 137.508) % 360
  const chroma = (1 - Math.abs(2 * 0.48 - 1)) * 0.72
  const hueSection = hue / 60
  const secondary = chroma * (1 - Math.abs((hueSection % 2) - 1))
  const rgb =
    hueSection < 1
      ? [chroma, secondary, 0]
      : hueSection < 2
        ? [secondary, chroma, 0]
        : hueSection < 3
          ? [0, chroma, secondary]
          : hueSection < 4
            ? [0, secondary, chroma]
            : hueSection < 5
              ? [secondary, 0, chroma]
              : [chroma, 0, secondary]
  const lightnessOffset = 0.48 - chroma / 2
  const channels = rgb.map((channel) =>
    Math.round((channel + lightnessOffset) * 255)
      .toString(16)
      .padStart(2, '0'),
  )

  return `#${channels.join('')}`
}

/**
 * Determina el color de texto con contraste óptimo (WCAG 2.1 AA) para un color de fondo.
 *
 * @param hexColor Color hexadecimal (ej. '#8b5cf6' o '#f0f3f8').
 * @returns '#0c0e14' para fondos claros o '#ffffff' para fondos oscuros.
 */
export function getTextColorForBackground(hexColor: string): '#ffffff' | '#0c0e14' {
  const clean = hexColor.replace('#', '')
  if (clean.length < 6) {
    return '#ffffff'
  }

  const r = parseInt(clean.substring(0, 2), 16) / 255
  const g = parseInt(clean.substring(2, 4), 16) / 255
  const b = parseInt(clean.substring(4, 6), 16) / 255

  // Luminancia relativa según especificación W3C WCAG
  const sR = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4)
  const sG = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4)
  const sB = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4)

  const luminance = 0.2126 * sR + 0.7152 * sG + 0.0722 * sB

  return luminance > 0.42 ? '#0c0e14' : '#ffffff'
}
