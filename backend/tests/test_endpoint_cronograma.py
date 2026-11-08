"""
test_endpoint_cronograma.py — Pruebas unitarias para el endpoint POST /cronograma (Ticket T18).
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


class TestEndpointCronogramaSuccess:
    """Pruebas de caso de éxito e integridad del cronograma retornado."""

    def test_cronograma_success_default_umbral(self):
        """POST /cronograma con umbral por defecto (15.0) debe devolver estructura completa y válida."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 200

        data = response.json()
        assert data["umbral"] == 15.0
        assert data["cantidad_nodos"] == 40
        assert len(data["nodos"]) == 40
        assert data["cantidad_turnos"] > 0
        assert data["cantidad_aristas"] > 0
        assert data["es_valido"] is True

        # Verificar que todos los nodos tengan turno asignado
        turnos_asignados = {nodo["id"]: nodo["turno"] for nodo in data["nodos"]}
        assert len(turnos_asignados) == 40
        for tid in turnos_asignados.values():
            assert isinstance(tid, int)
            assert tid >= 0

        # Verificar presencia y congruencia del agrupamiento por turno
        assert "turnos" in data
        assert isinstance(data["turnos"], dict)
        total_en_turnos = sum(len(materias) for materias in data["turnos"].values())
        assert total_en_turnos == 40

    def test_cronograma_no_adjacent_same_turn(self):
        """
        Regla de factibilidad académica fundamental:
        Ningún par de materias conectadas por una arista de conflicto puede compartir turno.
        """
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 200

        data = response.json()
        turnos_map = {n["id"]: n["turno"] for n in data["nodos"]}

        # Iterar sobre todas las aristas y asegurar que turnos difieran
        for arista in data["aristas"]:
            u = arista["source"]
            v = arista["target"]
            peso = arista["peso"]

            assert u in turnos_map, f"Materia {u} no encontrada en nodos"
            assert v in turnos_map, f"Materia {v} no encontrada en nodos"
            assert turnos_map[u] != turnos_map[v], (
                f"Conflicto de examen: materias adyacentes {u} y {v} "
                f"(peso {peso}) tienen el mismo turno ({turnos_map[u]})"
            )

    def test_cronograma_default_request_body_omitted(self):
        """Si se envía un cuerpo vacío '{}', debe tomar el umbral por defecto (15.0)."""
        response = client.post("/cronograma", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["umbral"] == 15.0
        assert data["es_valido"] is True


class TestEndpointCronogramaMonotonicityAndEdgeCases:
    """Pruebas de monotonía y límites extremos del grafo."""

    def test_cronograma_monotonicity(self):
        """
        Propiedad de monotonía: a mayor umbral τ, menor o igual cantidad de aristas
        y menor o igual cantidad de turnos requeridos.
        """
        r_10 = client.post("/cronograma", json={"umbral": 10.0})
        r_25 = client.post("/cronograma", json={"umbral": 25.0})
        r_50 = client.post("/cronograma", json={"umbral": 50.0})

        assert r_10.status_code == 200
        assert r_25.status_code == 200
        assert r_50.status_code == 200

        d_10 = r_10.json()
        d_25 = r_25.json()
        d_50 = r_50.json()

        # Monotonía en aristas
        assert d_10["cantidad_aristas"] >= d_25["cantidad_aristas"] >= d_50["cantidad_aristas"]

        # Monotonía en turnos necesarios
        assert d_10["cantidad_turnos"] >= d_25["cantidad_turnos"] >= d_50["cantidad_turnos"]

    def test_cronograma_empty_graph_one_turn(self):
        """
        Con un umbral gigante (ej. 400.0) que supera todo conflicto predicho,
        el grafo no tiene aristas y todas las materias deben poder rendirse en el turno 0 (1 turno total).
        """
        response = client.post("/cronograma", json={"umbral": 400.0})
        assert response.status_code == 200

        data = response.json()
        assert data["cantidad_aristas"] == 0
        assert data["aristas"] == []
        assert data["cantidad_turnos"] == 1
        assert data["cantidad_nodos"] == 40
        assert data["es_valido"] is True

        # Todas las materias asignadas al turno 0
        for n in data["nodos"]:
            assert n["turno"] == 0


class TestEndpointCronogramaValidationErrors:
    """Pruebas de validación de entradas y respuestas HTTP 422."""

    def test_cronograma_negative_umbral_returns_422(self):
        """Un umbral negativo debe ser rechazado con HTTP 422 Unprocessable Entity."""
        response = client.post("/cronograma", json={"umbral": -5.0})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_cronograma_excessive_umbral_returns_422(self):
        """Un umbral superior al límite máximo admitido (500.0) debe retornar HTTP 422."""
        response = client.post("/cronograma", json={"umbral": 600.0})
        assert response.status_code == 422

    def test_cronograma_invalid_type_returns_422(self):
        """Un valor no numérico debe retornar HTTP 422."""
        response = client.post("/cronograma", json={"umbral": "invalido"})
        assert response.status_code == 422

    def test_modelo_o_datos_no_encontrados_retorna_503(self, monkeypatch):
        """La falta de recursos para predecir debe comunicarse como HTTP 503."""
        from app import main as main_module

        def fail_predictions():
            raise FileNotFoundError("modelo.joblib")

        monkeypatch.setattr(main_module, "get_cached_predictions", fail_predictions)

        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 503
        assert "no se encontró el modelo o los datos requeridos" in response.json()["detail"]


class TestCronogramaOpenApiDocumentation:
    """Verifica que /cronograma figure registrado en la especificación OpenAPI."""

    def test_cronograma_in_openapi_docs(self):
        """El endpoint POST /cronograma debe figurar en /openapi.json con tag 'cronograma'."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        openapi = response.json()
        paths = openapi.get("paths", {})
        assert "/cronograma" in paths
        assert "post" in paths["/cronograma"]
        assert "cronograma" in paths["/cronograma"]["post"]["tags"]
