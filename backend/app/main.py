# para levantar el servidor: uvicorn app.main:app --reload
# (correrlo desde la carpeta /backend, no desde la raíz del repo)
# la documentación queda en http://localhost:8000/docs — ahí podés probar los endpoints directo

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# creamos la app, el title y description aparecen en /docs
app = FastAPI(
    title="Planificación de Mesas de Examen",
    description="API para el TP de grafos — Aviani & Canal",
    version="0.1.0",
)

# CORS: le decimos a fastapi que acepte pedidos desde el frontend
# si no ponés esto, el browser bloquea todo y te volvés loco debuggeando xd
# 5173 es el puerto default de Vite, si lo cambiás acordate de cambiarlo acá también
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],   # acepta GET, POST, etc.
    allow_headers=["*"],   # acepta cualquier header
)


@app.get("/", tags=["utils"])
def root():
    """Endpoint raíz con mensaje de bienvenida y estado de la API."""
    return {"status": "ok", "mensaje": "API de planificación de mesas de examen"}


# endpoint de salud — sirve para verificar que el server está vivo
# útil cuando no sabés si el problema es el backend o el frontend
@app.get("/health", tags=["utils"])
def health_check():
    """Endpoint de salud para monitoreo y verificación de estado."""
    # devuelve un json simple, nada más
    return {"status": "ok", "mensaje": "el servidor está vivo"}
