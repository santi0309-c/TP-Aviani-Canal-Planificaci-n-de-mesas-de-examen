"""Pruebas del endpoint GET /metricas (Ticket T19)."""

import json

from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app

client = TestClient(app)


def test_get_metricas_returns_test_metrics():
    """El endpoint debe devolver todos los datos del archivo de métricas."""
    expected = json.loads(main_module.METRICAS_JSON_PATH.read_text(encoding="utf-8"))

    response = client.get("/metricas")

    assert response.status_code == 200
    assert response.json() == expected


def test_get_metricas_returns_503_when_file_is_missing(tmp_path, monkeypatch):
    """Si todavía no se generaron métricas, el endpoint debe responder 503."""
    monkeypatch.setattr(
        main_module,
        "METRICAS_JSON_PATH",
        tmp_path / "metricas.json",
    )

    response = client.get("/metricas")

    assert response.status_code == 503
    assert response.json()["detail"] == "Las métricas todavía no están disponibles."


def test_get_metricas_returns_500_when_json_is_invalid(tmp_path, monkeypatch):
    """Un archivo que no contiene JSON válido debe comunicarse como error de servidor."""
    metrics_path = tmp_path / "metricas.json"
    metrics_path.write_text("{invalid json", encoding="utf-8")
    monkeypatch.setattr(main_module, "METRICAS_JSON_PATH", metrics_path)

    response = client.get("/metricas")

    assert response.status_code == 500
    assert response.json()["detail"] == "El archivo de métricas contiene JSON inválido."


def test_get_metricas_returns_500_when_data_does_not_match_schema(tmp_path, monkeypatch):
    """Un JSON válido con estructura incorrecta debe responder con error explícito."""
    metrics_path = tmp_path / "metricas.json"
    metrics_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main_module, "METRICAS_JSON_PATH", metrics_path)

    response = client.get("/metricas")

    assert response.status_code == 500
    assert response.json()["detail"] == "El archivo de métricas no cumple con el formato esperado."


def test_get_metricas_appears_in_openapi():
    """GET /metricas debe publicarse en la documentación OpenAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/metricas"]["get"]
    assert "metricas" in operation["tags"]
    assert "200" in operation["responses"]
    assert "500" in operation["responses"]
    assert "503" in operation["responses"]
