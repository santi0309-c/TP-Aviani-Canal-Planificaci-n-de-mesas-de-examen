// Calculamos un color para cada turno usando el ángulo áureo (137.508°)
// en el espacio HSL. De esta forma turnos consecutivos tienen colores
// bien distintos y el turno 0 siempre tiene el mismo color tanto en el
// grafo como en la grilla de materias.
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

// Luminancia relativa según la fórmula de accesibilidad WCAG 2.1
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

// Relación de contraste entre dos luminancias (fórmula W3C)
export function getContrastRatio(l1: number, l2: number): number {
  const lighter = Math.max(l1, l2)
  const darker = Math.min(l1, l2)
  return (lighter + 0.05) / (darker + 0.05)
}

// Elegimos si conviene texto blanco o negro sobre el fondo del turno.
// Calculamos el contraste real contra ambos y elegimos el que cumple
// WCAG AA (mínimo 4.5:1) para que se lea bien.
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
