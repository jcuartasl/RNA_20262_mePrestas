"""Pruebas de src/evaluation/distributions.py."""
import numpy as np
import pytest

from src.evaluation import distributions as d


@pytest.fixture
def calibrado():
    """p = E[y | p] por construcción: y = p + ruido independiente."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.3, 0.9, size=200_000)
    y = p + rng.normal(0, 0.15, size=p.size)
    return y, p


def test_distribuciones_identicas():
    rng = np.random.default_rng(2)
    y = rng.uniform(size=1000)
    res = d.comparar_distribuciones(y, y.copy())
    assert res["ks_2muestras"] == 0.0 and res["wasserstein"] == 0.0
    assert res["cociente_sd"] == pytest.approx(1.0)
    assert res["obs_bajo_min_pred"] == 0.0 and res["obs_sobre_max_pred"] == 0.0


def test_predictor_calibrado_cociente_sd_es_raiz_r2(calibrado):
    """Para la media condicional, Var(p) = R^2 Var(y): la compresión es esperada."""
    y, p = calibrado
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    res = d.comparar_distribuciones(y, p)
    assert res["cociente_sd"] == pytest.approx(np.sqrt(r2), abs=0.01)
    assert res["cociente_sd"] < 1


def test_calibracion_lineal(calibrado):
    y, p = calibrado
    res = d.calibracion_lineal(y, p)
    assert res["pendiente"] == pytest.approx(1.0, abs=0.01)
    assert res["intercepto"] == pytest.approx(0.0, abs=0.01)
    media = p.mean()
    disperso = media + 1.5 * (p - media)        # predicciones demasiado separadas
    assert d.calibracion_lineal(y, disperso)["pendiente"] < 1
    comprimido = media + 0.5 * (p - media)      # predicciones demasiado conservadoras
    assert d.calibracion_lineal(y, comprimido)["pendiente"] > 1


def test_calibracion_deciles(calibrado):
    y, p = calibrado
    tabla = d.calibracion_deciles(y, p, n_bins=10)
    assert len(tabla) == 10 and tabla["n"].sum() == y.size
    assert np.allclose(tabla["pred_media"], tabla["obs_media"], atol=0.01)


def test_tabla_cuantiles_y_figura(calibrado, tmp_path):
    y, p = calibrado
    tabla = d.tabla_cuantiles(y, {"modelo": p})
    assert list(tabla.columns) == ["observada", "modelo"]
    assert tabla["observada"].is_monotonic_increasing
    ruta = d.figura_distribuciones(y[:5000], {"modelo": p[:5000]}, tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists() and ruta == tmp_path / "fig.png"


def test_tamanos_distintos_fallan():
    with pytest.raises(ValueError):
        d.comparar_distribuciones([0.1, 0.2], [0.1])
