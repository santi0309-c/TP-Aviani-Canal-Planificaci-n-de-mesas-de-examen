"""
test_main.py — Tests para la API base de FastAPI (Ticket T02).
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Verifica que el endpoint raíz responda 200 con el mensaje esperado."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "mensaje" in data


def test_health_check():
    """Verifica que el endpoint /health responda 200 indicando que el servidor está vivo."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mensaje"] == "el servidor está vivo"


def test_cors_headers():
    """Verifica que CORS permita solicitudes desde el origen del frontend (Vite)."""
    response = client.get(
        "/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
