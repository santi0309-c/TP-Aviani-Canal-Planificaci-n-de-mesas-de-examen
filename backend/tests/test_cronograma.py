"""
test_cronograma.py — Pruebas unitarias para el endpoint POST /cronograma (Ticket T15).

Verifica la integración completa del pipeline:
    predicciones (T12) → grafo (T13) → coloreo greedy (T14) → cronograma (T15).
"""

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app, clear_api_cache

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_cache():
    """Limpia las cachés antes y después de cada test para evitar interferencias."""
    clear_api_cache()
    yield
    clear_api_cache()


class TestCronogramaEstructura:
    """Verifica la estructura básica de la respuesta del endpoint."""

    def test_cronograma_status_200(self):
        """POST /cronograma debe responder HTTP 200."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.status_code == 200

    def test_cronograma_campos_obligatorios(self):
        """La respuesta debe contener todos los campos del CronogramaResponse."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        data = response.json()

        expected_keys = {
            "umbral",
            "cantidad_turnos",
            "cantidad_nodos",
            "cantidad_aristas",
            "es_coloreo_valido",
            "turnos",
            "nodos",
            "aristas",
        }
        assert expected_keys.issubset(data.keys())

    def test_cronograma_umbral_en_respuesta(self):
        """El umbral devuelto debe coincidir con el enviado."""
        response = client.post("/cronograma", json={"umbral": 20.0})
        assert response.json()["umbral"] == 20.0

    def test_cronograma_umbral_default(self):
        """Sin indicar umbral, debe usarse 15.0 como default."""
        response = client.post("/cronograma", json={})
        assert response.status_code == 200
        assert response.json()["umbral"] == 15.0


class TestCronogramaColoreo:
    """Verifica la corrección matemática del coloreo greedy devuelto."""

    def test_coloreo_valido(self):
        """es_coloreo_valido debe ser True: ninguna arista puede tener extremos del mismo turno."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        assert response.json()["es_coloreo_valido"] is True

    def test_coloreo_valido_umbral_bajo(self):
        """El coloreo debe seguir siendo válido con un umbral bajo (más aristas)."""
        response = client.post("/cronograma", json={"umbral": 5.0})
        assert response.json()["es_coloreo_valido"] is True

    def test_todas_las_materias_tienen_turno(self):
        """Cada nodo en la respuesta debe tener un turno asignado (entero >= 0)."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        nodos = response.json()["nodos"]
        assert len(nodos) > 0
        for nodo in nodos:
            assert "turno" in nodo
            assert isinstance(nodo["turno"], int)
            assert nodo["turno"] >= 0

    def test_turnos_agrupados_cubren_todas_las_materias(self):
        """La suma de materias en turnos debe igualar la cantidad de nodos."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        data = response.json()

        total_en_turnos = sum(len(mats) for mats in data["turnos"].values())
        assert total_en_turnos == data["cantidad_nodos"]

    def test_aristas_no_comparten_turno(self):
        """Para cada arista en la respuesta, source y target deben tener turnos distintos."""
        response = client.post("/cronograma", json={"umbral": 15.0})
        data = response.json()

        # Construir mapa nodo -> turno
        turno_map = {n["id"]: n["turno"] for n in data["nodos"]}

        for arista in data["aristas"]:
            src_turno = turno_map.get(arista["source"])
            tgt_turno = turno_map.get(arista["target"])
            assert src_turno is not None
            assert tgt_turno is not None
            assert src_turno != tgt_turno, (
                f"Conflicto de coloreo: {arista['source']} y {arista['target']} "
                f"comparten el turno {src_turno}"
            )


class TestCronogramaMonotonia:
    """Verifica que el comportamiento sea consistente con el trade-off de umbral."""

    def test_umbral_alto_menos_aristas(self):
        """Un umbral más alto debe producir igual o menos aristas que uno bajo."""
        r_low = client.post("/cronograma", json={"umbral": 5.0})
        r_high = client.post("/cronograma", json={"umbral": 50.0})

        assert r_low.status_code == 200
        assert r_high.status_code == 200

        aristas_low = r_low.json()["cantidad_aristas"]
        aristas_high = r_high.json()["cantidad_aristas"]

        assert aristas_low >= aristas_high

    def test_umbral_alto_no_mas_turnos(self):
        """Un umbral más alto (menos aristas) no puede requerir más turnos que uno bajo."""
        r_low = client.post("/cronograma", json={"umbral": 5.0})
        r_high = client.post("/cronograma", json={"umbral": 50.0})

        turnos_low = r_low.json()["cantidad_turnos"]
        turnos_high = r_high.json()["cantidad_turnos"]

        assert turnos_low >= turnos_high

    def test_umbral_extremo_grafo_vacio(self):
        """Con umbral muy alto (ej. 400), no debe haber aristas y el coloreo usa 1 turno."""
        response = client.post("/cronograma", json={"umbral": 400.0})
        data = response.json()

        assert data["cantidad_aristas"] == 0
        # Grafo sin aristas: todas las materias van al mismo turno (turno 0)
        assert data["cantidad_turnos"] == 1
        assert data["es_coloreo_valido"] is True

    def test_grafo_vacio_un_turno(self):
        """Sin aristas, todas las materias deben estar en el turno 0."""
        response = client.post("/cronograma", json={"umbral": 400.0})
        data = response.json()

        assert data["cantidad_turnos"] == 1
        assert "0" in data["turnos"]
        assert len(data["turnos"]["0"]) == data["cantidad_nodos"]


class TestCronogramaValidacionInput:
    """Verifica la validación del cuerpo del request."""

    def test_umbral_negativo_rechazado(self):
        """Un umbral negativo debe retornar HTTP 422."""
        response = client.post("/cronograma", json={"umbral": -1.0})
        assert response.status_code == 422

    def test_umbral_excesivo_rechazado(self):
        """Un umbral > 500 debe retornar HTTP 422."""
        response = client.post("/cronograma", json={"umbral": 600.0})
        assert response.status_code == 422

    def test_modelo_o_datos_no_encontrados_retorna_503(self, monkeypatch):
        """La falta de recursos para predecir debe comunicarse como HTTP 503."""
        def fail_predictions():
            raise FileNotFoundError("modelo.joblib")

        monkeypatch.setattr(main_module, "get_cached_predictions", fail_predictions)

        response = client.post("/cronograma", json={"umbral": 15.0})

        assert response.status_code == 503
        assert "falta el modelo" in response.json()["detail"]

    def test_cronograma_registrado_en_openapi(self):
        """El path /cronograma debe figurar registrado en /openapi.json."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()
        paths = openapi.get("paths", {})
        assert "/cronograma" in paths
        assert "post" in paths["/cronograma"]
        assert "cronograma" in paths["/cronograma"]["post"]["tags"]
