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
 * Calcula la luminancia relativa de un color sRGB según la especificación W3C WCAG 2.1.
 *
 * @param hexColor Color en formato hexadecimal (ej. '#8b5cf6' o '#22b5d3').
 * @returns Luminancia relativa en el rango [0, 1].
 */
export function getRelativeLuminance(hexColor: string): number {
  const clean = hexColor.replace('#', '')
  if (clean.length < 6) {
    return 0
  }

  const r = parseInt(clean.substring(0, 2), 16) / 255
  const g = parseInt(clean.substring(2, 4), 16) / 255
  const b = parseInt(clean.substring(4, 6), 16) / 255

  // Conversión gamma sRGB a espacio lineal
  const sR = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4)
  const sG = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4)
  const sB = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4)

  return 0.2126 * sR + 0.7152 * sG + 0.0722 * sB
}

/**
 * Calcula la relación de contraste (Contrast Ratio) entre dos luminancias relativas según WCAG 2.1.
 * Fórmula W3C: (L1 + 0.05) / (L2 + 0.05), donde L1 es la luminancia más clara.
 *
 * @param l1 Luminancia relativa del primer color [0, 1].
 * @param l2 Luminancia relativa del segundo color [0, 1].
 * @returns Relación de contraste en escala 1 a 21 (ej. 4.5 para 4.5:1).
 */
export function getContrastRatio(l1: number, l2: number): number {
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

/**
 * Determina el color de texto con contraste que garantiza cumplimiento WCAG 2.1 AA (mínimo 4.5:1).
 *
 * En lugar de usar un umbral fijo arbitrario de luminancia, calcula la relación de contraste
 * real según la fórmula de la W3C frente a opciones de texto ('#ffffff' y '#000000') y selecciona
 * la que cumpla el estándar WCAG AA (>= 4.5:1) maximizando la legibilidad perceptual.
 *
 * @param hexColor Color de fondo hexadecimal (ej. '#8b5cf6' o '#22b5d3').
 * @returns '#ffffff' para fondos oscuros o '#000000' para fondos claros.
 */
export function getTextColorForBackground(hexColor: string): '#ffffff' | '#000000' {
  const bgLuminance = getRelativeLuminance(hexColor)
  const whiteLuminance = 1.0 // Luminancia relativa de #ffffff
  const blackLuminance = 0.0 // Luminancia relativa de #000000

  const contrastWithWhite = getContrastRatio(bgLuminance, whiteLuminance)
  const contrastWithBlack = getContrastRatio(bgLuminance, blackLuminance)

  // Priorizar la opción que supere 4.5:1 (WCAG AA); si ambas superan, elegir la de mayor contraste
  if (contrastWithWhite >= 4.5 && contrastWithBlack < 4.5) {
    return '#ffffff'
  }
  if (contrastWithBlack >= 4.5 && contrastWithWhite < 4.5) {
    return '#000000'
  }
  return contrastWithWhite >= contrastWithBlack ? '#ffffff' : '#000000'
}
