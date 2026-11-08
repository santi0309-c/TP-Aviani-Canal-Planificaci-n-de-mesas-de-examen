"""
test_endpoints_lectura.py — Pruebas unitarias para los endpoints de lectura /materias y /conflictos (Ticket T17).
"""

from fastapi.testclient import TestClient
import pytest

from app.main import app, clear_api_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_api_cache():
    """Limpia las cachés antes y después de cada test."""
    clear_api_cache()
    yield
    clear_api_cache()


class TestGetMaterias:
    """Pruebas para el endpoint GET /materias."""

    def test_get_materias_status_and_count(self):
        """GET /materias debe responder HTTP 200 y devolver exactamente 40 materias."""
        response = client.get("/materias")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 40

    def test_get_materias_structure(self):
        """Cada materia debe tener los atributos tipados según MateriaSchema."""
        response = client.get("/materias")
        assert response.status_code == 200

        data = response.json()
        for materia in data:
            assert "id" in materia
            assert "nombre" in materia
            assert "anio" in materia
            assert "carreras" in materia
            assert "tamanio" in materia

            assert isinstance(materia["id"], str)
            assert isinstance(materia["nombre"], str)
            assert 1 <= materia["anio"] <= 5
            assert isinstance(materia["carreras"], list)
            assert len(materia["carreras"]) >= 1
            assert isinstance(materia["tamanio"], int)
            assert materia["tamanio"] >= 0

        # Verificar presencia de materias clave de primer año y tronco común
        ids = {m["id"] for m in data}
        assert "M01" in ids
        assert "M40" in ids


class TestGetConflictos:
    """Pruebas para el endpoint GET /conflictos."""

    def test_get_conflictos_default_umbral(self):
        """GET /conflictos sin parámetros debe usar umbral 15.0 por defecto."""
        response = client.get("/conflictos")
        assert response.status_code == 200

        data = response.json()
        assert "umbral" in data
        assert "total_conflictos" in data
        assert "conflictos" in data

        assert data["umbral"] == 15.0
        assert isinstance(data["conflictos"], list)
        assert data["total_conflictos"] == len(data["conflictos"])
        assert data["total_conflictos"] > 0

        # Verificar que todos los conflictos retornados cumplan con la cota
        for item in data["conflictos"]:
            assert item["conflicto_predicho"] >= 15.0
            assert "materia_1" in item
            assert "materia_2" in item

    def test_get_conflictos_monotonicity(self):
        """
        Propiedad de monotonía: a mayor umbral, la cantidad de aristas/conflictos
        debe ser menor o igual (nunca mayor).
        """
        r_10 = client.get("/conflictos?umbral=10.0")
        r_25 = client.get("/conflictos?umbral=25.0")
        r_50 = client.get("/conflictos?umbral=50.0")

        assert r_10.status_code == 200
        assert r_25.status_code == 200
        assert r_50.status_code == 200

        c_10 = r_10.json()["total_conflictos"]
        c_25 = r_25.json()["total_conflictos"]
        c_50 = r_50.json()["total_conflictos"]

        assert c_10 >= c_25 >= c_50

    def test_get_conflictos_high_umbral_empty(self):
        """Un umbral suficientemente alto (ej. 300) no debe generar aristas."""
        response = client.get("/conflictos?umbral=300.0")
        assert response.status_code == 200

        data = response.json()
        assert data["umbral"] == 300.0
        assert data["total_conflictos"] == 0
        assert data["conflictos"] == []

    def test_get_conflictos_zero_umbral_returns_all_or_most(self):
        """Un umbral 0.0 debe devolver todos los pares con conflicto >= 0."""
        response = client.get("/conflictos?umbral=0.0")
        assert response.status_code == 200

        data = response.json()
        # Casi todos los pares tendrán predicción >= 0
        assert data["total_conflictos"] >= 700

    def test_get_conflictos_invalid_umbral_negative(self):
        """Un umbral negativo debe retornar HTTP 422 Unprocessable Entity."""
        response = client.get("/conflictos?umbral=-5")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_get_conflictos_invalid_umbral_type(self):
        """Un umbral con tipo no numérico debe retornar HTTP 422."""
        response = client.get("/conflictos?umbral=invalido")
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_get_conflictos_exceeding_max_umbral(self):
        """Un umbral que exceda el límite máximo le=500.0 debe retornar HTTP 422."""
        response = client.get("/conflictos?umbral=600.0")
        assert response.status_code == 422


class TestOpenApiDocumentation:
    """Verifica que los nuevos endpoints aparezcan documentados en OpenAPI / Swagger."""

    def test_docs_and_openapi_registered(self):
        """Los paths /materias y /conflictos deben figurar registrados en /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi.get("paths", {})

        assert "/materias" in paths
        assert "get" in paths["/materias"]
        assert "materias" in paths["/materias"]["get"]["tags"]

        assert "/conflictos" in paths
        assert "get" in paths["/conflictos"]
        assert "conflictos" in paths["/conflictos"]["get"]["tags"]

