"""
main.py — Punto de entrada de la API (FastAPI).
TODO: implementar los endpoints /materias, /conflictos, /cronograma, /metricas
"""
from fastapi import FastAPI

app = FastAPI(title="Planificación de Mesas de Examen", version="0.1.0")


@app.get("/")
def root():
    return {"status": "ok", "mensaje": "API de planificación de mesas de examen"}
