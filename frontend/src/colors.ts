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

