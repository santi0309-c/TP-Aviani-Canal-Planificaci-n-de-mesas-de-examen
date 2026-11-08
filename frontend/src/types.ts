export interface HealthResponse {
  status: string
  mensaje: string
}

export interface Materia {
  id: string
  nombre: string
  anio: number
  carreras: string[]
  tamanio: number
}

export interface Conflicto {
  materia_1: string
  materia_2: string
  conflicto_predicho: number
  cant_estudiantes_compartidos_historico?: number | null
  mismo_anio?: boolean | null
  misma_carrera?: boolean | null
}

export interface ConflictosResponse {
  umbral: number
  total_conflictos: number
  conflictos: Conflicto[]
}

export interface NodoCronograma {
  id: string
  nombre: string
  turno: number
  carreras?: string[] | null
  anio?: number | null
}

export interface AristaCronograma {
  source: string
  target: string
  peso: number
}

export interface CronogramaRequest {
  umbral?: number
}

export interface CronogramaResponse {
  umbral: number
  cantidad_turnos: number
  cantidad_nodos: number
  cantidad_aristas: number
  nodos: NodoCronograma[]
  aristas: AristaCronograma[]
  turnos?: Record<string, string[]> | null
  es_valido: boolean
  es_coloreo_valido?: boolean | null
}

export interface MetricasMetadata {
  modelo: string
  semilla: number
  n_test: number
}

export interface MetricasRegresion {
  mae: number
  rmse: number
  r2: number
}

export interface MetricasClasificacion {
  umbral: number
  accuracy: number
  precision: number
  recall: number
  f1: number
}

export interface MatrizConfusion {
  verdaderos_negativos: number
  falsos_positivos: number
  falsos_negativos: number
  verdaderos_positivos: number
  total: number
}

export interface MetricasResponse {
  metadatos: MetricasMetadata
  regresion: MetricasRegresion
  clasificacion: MetricasClasificacion
  matriz_confusion: MatrizConfusion
}
