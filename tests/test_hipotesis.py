"""Pruebas de las funciones estadísticas del contraste de hipótesis."""
import numpy as np
import pytest

from src.interpret.hipotesis import comparar_medias, comparar_tasas, prueba_no_linealidad


def test_comparar_tasas_valores_conocidos():
    r = comparar_tasas(np.r_[np.ones(300), np.zeros(700)], np.r_[np.ones(200), np.zeros(800)])
    assert r["valor_a"] == 0.3 and r["valor_b"] == 0.2 and r["razon"] == pytest.approx(1.5)
    assert r["ic_inf"] < 1.5 < r["ic_sup"] and r["p_valor"] < 1e-5


def test_comparar_tasas_sin_efecto():
    rng = np.random.default_rng(0)
    r = comparar_tasas(rng.uniform(size=5000) < 0.2, rng.uniform(size=5000) < 0.2)
    assert r["ic_inf"] < 1 < r["ic_sup"] and r["p_valor"] > 0.01


def test_comparar_medias():
    rng = np.random.default_rng(1)
    r = comparar_medias(rng.normal(0.8, 0.1, 2000), rng.normal(0.7, 0.1, 2000))
    assert r["diferencia"] == pytest.approx(0.1, abs=0.01) and r["p_valor"] < 1e-10


def test_no_linealidad_detecta_curvatura_y_no_inventa():
    rng = np.random.default_rng(2)
    x = rng.uniform(-2, 2, 20000)
    y_lineal = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-x))).astype(float)
    y_curvo = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(x ** 2 - 1)))).astype(float)
    assert prueba_no_linealidad(x, y_curvo)["p_valor"] < 1e-6
    assert prueba_no_linealidad(x, y_lineal)["p_valor"] > 0.01
