"""
test_eda.py — Smoke tests para el script de análisis exploratorio (Ticket T06).
"""

import os
import tempfile
import pytest

from data.eda import DEFAULT_DATA, run_eda


def test_run_eda_generates_all_plots():
    """Verifica que run_eda genere correctamente los 4 gráficos requeridos."""
    assert os.path.exists(DEFAULT_DATA), f"No existe el archivo de datos: {DEFAULT_DATA}"

    with tempfile.TemporaryDirectory() as tmpdir:
        run_eda(data_path=DEFAULT_DATA, out_dir=tmpdir)

        expected_plots = [
            "01_distribucion_targets.png",
            "02_balance_clases_conflicto.png",
            "03_matriz_correlacion.png",
            "04_dispersion_afectados_vs_compartidos.png",
        ]

        for plot_name in expected_plots:
            plot_path = os.path.join(tmpdir, plot_name)
            assert os.path.isfile(plot_path), f"El gráfico no fue creado: {plot_name}"
            assert os.path.getsize(plot_path) > 0, f"El gráfico generado está vacío: {plot_name}"


def test_run_eda_missing_dataset_raises():
    """Verifica que run_eda lance FileNotFoundError ante una ruta inexistente."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_path = os.path.join(tmpdir, "inexistente.csv")
        with pytest.raises(FileNotFoundError):
            run_eda(data_path=fake_path, out_dir=tmpdir)
