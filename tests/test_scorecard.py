"""Pruebas de src/scorecard/scorecard.py: propiedades que debe cumplir toda scorecard."""
import numpy as np
import pytest

from src.scorecard.scorecard import Scorecard, percentil

SC = Scorecard(600, 50, 20)


def test_factor_y_offset():
    assert SC.factor == pytest.approx(20 / np.log(2))
    assert SC.offset == pytest.approx(600 - 20 / np.log(2) * np.log(50))


def test_odds_base_da_puntaje_base():
    assert SC.puntaje_desde_pd(1 / 51) == pytest.approx(600)


def test_duplicar_odds_suma_pdo():
    odds = np.array([0.5, 1, 4, 50, 300])
    p, p_doble = 1 / (1 + odds), 1 / (1 + 2 * odds)
    assert np.allclose(SC.puntaje_desde_pd(p_doble) - SC.puntaje_desde_pd(p), 20)


def test_valores_conocidos():
    assert SC.puntaje_desde_pd(0.5) == pytest.approx(487.12, abs=0.01)
    assert SC.puntaje_desde_pd(0.2) == pytest.approx(527.12, abs=0.01)
    assert SC.puntaje_desde_pd(1 / 101) == pytest.approx(620)


def test_menor_pd_mas_puntaje():
    p = np.linspace(0.001, 0.999, 500)
    assert np.all(np.diff(SC.puntaje_desde_pd(p)) < 0)


def test_logit_y_pd_coinciden():
    z = np.linspace(-8, 8, 50)
    p = 1 / (1 + np.exp(-z))
    assert np.allclose(SC.puntaje_desde_logit(z), SC.puntaje_desde_pd(p))


def test_ida_y_vuelta():
    p = np.array([0.001, 0.02, 0.2, 0.5, 0.9])
    assert np.allclose(SC.pd_desde_puntaje(SC.puntaje_desde_pd(p)), p)


def test_cambiar_pdo_cambia_escala_no_orden():
    p = np.array([0.05, 0.1, 0.3])
    otra = Scorecard(600, 50, 40)
    assert np.all(np.diff(otra.puntaje_desde_pd(p)) < 0)
    assert otra.puntaje_desde_pd(1 / 51) == pytest.approx(600)
    assert otra.puntaje_desde_pd(1 / 101) == pytest.approx(640)


def test_desde_config():
    assert Scorecard.desde_config({"base_score": 600, "base_odds": 50, "pdo": 20}) == SC


def test_parametros_invalidos():
    with pytest.raises(ValueError):
        Scorecard(600, 0, 20)
    with pytest.raises(ValueError):
        Scorecard(600, 50, -20)


def test_percentil():
    ref = np.arange(1, 101)                          # puntajes 1..100
    assert percentil(0, ref) == 0
    assert percentil(51, ref) == 50
    assert percentil(1000, ref) == 100
    assert np.allclose(percentil([1, 101], ref), [0, 100])
    with pytest.raises(ValueError):
        percentil(10, [])
