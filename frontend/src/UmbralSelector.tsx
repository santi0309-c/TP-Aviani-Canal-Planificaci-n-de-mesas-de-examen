import { useCallback, useEffect, useState } from 'react'
import { useDebounce } from './useDebounce'

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

  // Sincronizar estado durante el render si cambia la prop externa (patrón oficial de React 19)
  if (value !== prevValue) {
    setPrevValue(value)
    setLocalValue(value)
  }

  const debouncedValue = useDebounce<number>(localValue, 350)

  // Propagar el valor estabilizado tras el debounce
  useEffect(() => {
    if (debouncedValue !== value) {
      onChange(debouncedValue)
    }
  }, [debouncedValue, value, onChange])

  const handleSliderChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const nextValue = Number(e.target.value)
      setLocalValue(nextValue)
    },
    [],
  )

  const handlePresetClick = useCallback(
    (presetValue: number) => {
      setLocalValue(presetValue)
      onChange(presetValue)
    },
    [onChange],
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
            <span className="umbral-spinner" title="Actualizando grafo..." />
          )}
        </div>
      </div>

      <div className="slider-wrapper">
        <input
          id="umbral-slider"
          type="range"
          min={min}
          max={max}
          step={step}
          value={localValue}
          onChange={handleSliderChange}
          disabled={disabled}
          className="umbral-range-slider"
          style={{
            background: `linear-gradient(to right, var(--accent-primary) 0%, var(--accent-primary) ${percentage}%, rgba(255, 255, 255, 0.08) ${percentage}%, rgba(255, 255, 255, 0.08) 100%)`,
          }}
          aria-label="Selector de umbral de conflicto para exámenes"
          aria-valuemin={min}
          aria-valuemax={max}
          aria-valuenow={localValue}
          aria-valuetext={`${localValue} estudiantes`}
        />
        <div className="slider-ticks" aria-hidden="true">
          <span>{min} (Todos en conflicto)</span>
          <span>50</span>
          <span>{max} (Sin aristas)</span>
        </div>
      </div>

      <div className="umbral-presets-wrapper">
        <span className="presets-label">Valores de referencia:</span>
        <div className="presets-buttons" role="group" aria-label="Presets de umbral">
          {PRESETS.map((p) => {
            const isSelected = localValue === p.value
            return (
              <button
                key={p.value}
                type="button"
                className={`btn-preset ${isSelected ? 'preset-active' : ''}`}
                onClick={() => handlePresetClick(p.value)}
                disabled={disabled}
                title={`${p.label} (${p.hint})`}
              >
                {p.label} <strong>({p.value})</strong>
              </button>
            )
          })}
        </div>
      </div>

      <div className="tradeoff-info" role="status" aria-live="polite">
        <span className="tradeoff-icon">ℹ</span>
        <p className="tradeoff-text">{getTradeOffDescription(localValue)}</p>
      </div>
    </div>
  )
}
