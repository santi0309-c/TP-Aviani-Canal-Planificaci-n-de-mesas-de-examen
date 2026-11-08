import { useCallback, useState } from 'react'
import { useDebouncedCallback } from './useDebounce'

export interface UmbralSelectorProps {
  value: number
  onChange: (value: number) => void
  disabled?: boolean
  min?: number
  max?: number
  step?: number
  isUpdating?: boolean
}

const PRESETS = [
  { label: 'Estricto', value: 5, hint: 'τ = 5' },
  { label: 'Recomendado', value: 15, hint: 'τ = 15' },
  { label: 'Moderado', value: 30, hint: 'τ = 30' },
  { label: 'Permisivo', value: 50, hint: 'τ = 50' },
] as const

function getTradeOffDescription(umbral: number): string {
  if (umbral <= 8) {
    return 'Alta restricción: Se detectan conflictos leves. Requiere más turnos de examen para evitar cualquier superposición.'
  }
  if (umbral <= 20) {
    return 'Equilibrio institucional óptimo: Balance ideal entre número de turnos de examen y prevención de conflictos.'
  }
  if (umbral <= 40) {
    return 'Tolerancia moderada: Se admiten cruces menores para compactar el calendario en menos turnos.'
  }
  return 'Baja restricción: Solo se restringen conflictos masivos. Número mínimo de turnos de examen.'
}

export function UmbralSelector({
  value,
  onChange,
  disabled = false,
  min = 0,
  max = 100,
  step = 1,
  isUpdating = false,
}: UmbralSelectorProps) {
  const [prevValue, setPrevValue] = useState<number>(value)
  const [localValue, setLocalValue] = useState<number>(value)

  const debouncedOnChange = useDebouncedCallback(onChange, 350)

  // Sincronizar estado durante el render si cambia la prop externa (patrón oficial de React 19)
  if (value !== prevValue) {
    setPrevValue(value)
    setLocalValue(value)
    debouncedOnChange.cancel()
  }

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const nextValue = Number(e.target.value)
      setLocalValue(nextValue)
      debouncedOnChange.run(nextValue)
    },
    [debouncedOnChange],
  )

  const handlePresetClick = useCallback(
    (presetValue: number) => {
      setLocalValue(presetValue)
      // Ejecutar inmediatamente y cancelar cualquier debounce pendiente del slider
      debouncedOnChange.flush(presetValue)
    },
    [debouncedOnChange],
  )

  const percentage = Math.min(
    100,
    Math.max(0, ((localValue - min) / (max - min)) * 100),
  )

  return (
    <div className="umbral-selector-container">
      <div className="umbral-header">
        <div className="umbral-label-group">
          <label htmlFor="umbral-slider" className="umbral-title">
            Umbral de conflicto (τ)
          </label>
          <span className="umbral-subtitle">
            Mínimo de estudiantes afectados para considerar conflicto
          </span>
        </div>

        <div className="umbral-value-display">
          <span
            className={`umbral-badge ${isUpdating ? 'badge-updating' : ''}`}
            title="Valor actual del umbral de corte"
          >
            τ = <strong>{localValue}</strong>
            <span className="umbral-unit">estudiantes</span>
          </span>

          {isUpdating && (
            <div
              className="umbral-spinner"
              title="Recalculando grafo y coloreo..."
              role="status"
              aria-label="Cargando"
            />
          )}
        </div>
      </div>

      {/* Slider interactivo */}
      <div className="slider-wrapper">
        <input
          id="umbral-slider"
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          disabled={disabled}
          onChange={handleSliderChange}
          className="umbral-range-slider"
          aria-label="Umbral de conflicto para considerar materias superpuestas"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={localValue}
          style={{
            background: `linear-gradient(to right, #8b5cf6 0%, #8b5cf6 ${percentage}%, rgba(255, 255, 255, 0.1) ${percentage}%, rgba(255, 255, 255, 0.1) 100%)`,
          }}
        />

        <div className="slider-ticks" aria-hidden="true">
          <span>{min} (Máx. conflictos)</span>
          <span>50</span>
          <span>{max} (Mín. conflictos)</span>
        </div>
      </div>

      {/* Presets de acceso rápido */}
      <div className="umbral-presets-wrapper">
        <span className="presets-label">Presets:</span>
        <div className="presets-buttons" role="group" aria-label="Presets de umbral">
          {PRESETS.map((preset) => {
            const isActive = localValue === preset.value
            return (
              <button
                key={preset.value}
                type="button"
                className={`btn-preset ${isActive ? 'preset-active' : ''}`}
                onClick={() => handlePresetClick(preset.value)}
                disabled={disabled}
                title={`Fijar umbral en ${preset.value} (${preset.label})`}
              >
                <span>{preset.label}</span>
                <code>{preset.hint}</code>
              </button>
            )
          })}
        </div>
      </div>

      {/* Tradeoff explicativo */}
      <div className="tradeoff-info" role="note">
        <span className="tradeoff-icon">ℹ</span>
        <p className="tradeoff-text">{getTradeOffDescription(localValue)}</p>
      </div>
    </div>
  )
}
