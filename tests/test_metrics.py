"""Pruebas de src/evaluation/metrics.py contra implementaciones de referencia y valores teóricos."""
import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)

from src.evaluation import metrics as m


@pytest.fixture
def datos_pd():
    rng = np.random.default_rng(123)
    p = rng.uniform(0.01, 0.99, size=3000)
    y = (rng.uniform(size=3000) < p).astype(int)
    return y, p


@pytest.fixture
def datos_pd_empates(datos_pd):
    y, p = datos_pd
    return y, np.round(p, 1)  # fuerza muchos empates


# ------------------------------------------------ contra referencias

@pytest.mark.parametrize("fixture", ["datos_pd", "datos_pd_empates"])
def test_auc_igual_a_sklearn(fixture, request):
    y, p = request.getfixturevalue(fixture)
    assert m.auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


@pytest.mark.parametrize("fixture", ["datos_pd", "datos_pd_empates"])
def test_ks_igual_a_scipy(fixture, request):
    y, p = request.getfixturevalue(fixture)
    assert m.ks(y, p) == pytest.approx(ks_2samp(p[y == 1], p[y == 0]).statistic, abs=1e-12)


def test_gini_es_2auc_menos_1(datos_pd):
    y, p = datos_pd
    assert m.gini(y, p) == pytest.approx(2 * roc_auc_score(y, p) - 1, abs=1e-12)


def test_brier_y_bce_iguales_a_sklearn(datos_pd):
    y, p = datos_pd
    assert m.brier(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    assert m.bce(y, p) == pytest.approx(log_loss(y, p), abs=1e-12)


def test_metricas_lgd_iguales_a_sklearn():
    rng = np.random.default_rng(7)
    y = rng.beta(0.5, 0.5, size=2000)
    p = np.clip(y + rng.normal(0, 0.15, size=2000), 0, 1)
    res = m.metricas_lgd(y, p)
    assert res["mae"] == pytest.approx(mean_absolute_error(y, p), abs=1e-12)
    assert res["rmse"] == pytest.approx(np.sqrt(mean_squared_error(y, p)), abs=1e-12)
    assert res["r2"] == pytest.approx(r2_score(y, p), abs=1e-12)


def test_metricas_lgd_coincide_con_version_provisional_de_p3():
    from src.models.lgd_mlp import metricas_lgd as provisional
    rng = np.random.default_rng(8)
    y, p = rng.uniform(size=500), rng.uniform(size=500)
    nuevo, viejo = m.metricas_lgd(y, p), provisional(y, p)
    for clave in ("rmse", "mae", "r2"):
        assert nuevo[clave] == pytest.approx(viejo[clave], abs=1e-12)


# ------------------------------------------------ valores teóricos

def test_valores_teoricos_uniforme_bernoulli():
    """p ~ U(0,1), y ~ Bernoulli(p): AUC = 5/6, KS = 1/2, Brier = 1/6, BCE = 1/2."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=400_000)
    y = (rng.uniform(size=400_000) < p).astype(int)
    res = m.metricas_pd(y, p)
    assert res["auc"] == pytest.approx(5 / 6, abs=5e-3)
    assert res["ks"] == pytest.approx(0.5, abs=5e-3)
    assert res["brier"] == pytest.approx(1 / 6, abs=5e-3)
    assert res["bce"] == pytest.approx(0.5, abs=5e-3)


# ------------------------------------------------ casos extremos y propiedades

def test_separacion_perfecta_e_invertida():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert m.auc(y, p) == 1.0 and m.ks(y, p) == 1.0 and m.gini(y, p) == 1.0
    assert m.auc(y, 1 - p) == 0.0


def test_puntaje_constante_no_discrimina():
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.3)
    assert m.auc(y, p) == 0.5 and m.ks(y, p) == 0.0


def test_auc_y_ks_invariantes_a_transformaciones_monotonas(datos_pd):
    """AUC y KS solo dependen del orden: valen igual sobre la PD, el logit o un puntaje."""
    y, p = datos_pd
    logit = np.log(p / (1 - p))
    assert m.auc(y, logit) == pytest.approx(m.auc(y, p), abs=1e-12)
    assert m.ks(y, logit) == pytest.approx(m.ks(y, p), abs=1e-12)


@pytest.mark.parametrize("estrategia", ["cuantil", "uniforme"])
def test_curva_calibracion_consistente(datos_pd, estrategia):
    y, p = datos_pd
    tabla = m.curva_calibracion(y, p, n_bins=10, estrategia=estrategia)
    assert tabla["n"].sum() == y.size
    assert len(tabla) <= 10
    tasa_global = (tabla["tasa_observada"] * tabla["n"]).sum() / tabla["n"].sum()
    assert tasa_global == pytest.approx(y.mean(), abs=1e-12)


def test_curva_calibracion_cuantil_bins_balanceados(datos_pd):
    y, p = datos_pd
    tabla = m.curva_calibracion(y, p, n_bins=10)
    assert tabla["n"].max() - tabla["n"].min() <= 1


# ------------------------------------------------ errores

def test_errores_de_entrada():
    with pytest.raises(ValueError):
        m.auc([1, 1, 1], [0.2, 0.5, 0.9])          # una sola clase
    with pytest.raises(ValueError):
        m.auc([0, 1], [0.2, 0.5, 0.9])             # tamaños distintos
    with pytest.raises(ValueError):
        m.auc([0, 2, 1], [0.2, 0.5, 0.9])          # y no binaria
    with pytest.raises(ValueError):
        m.brier([0, 1], [0.2, 1.5])                # p fuera de [0, 1]
    with pytest.raises(ValueError):
        m.rmse([0.1, np.nan], [0.2, 0.3])          # NaN
    with pytest.raises(ValueError):
        m.r2([0.5, 0.5], [0.2, 0.3])               # y constante
    with pytest.raises(ValueError):
        m.curva_calibracion([0, 1], [0.2, 0.8], estrategia="otra")
