"""Pruebas de la API de la scorecard con los artefactos reales del repositorio."""
import numpy as np
import pandas as pd
import pytest

from src.data import load_splits
from src.models.pd_logreg import cargar_logistica
from src.scorecard.api import ScorecardAPI
from src.scorecard.scorecard import Scorecard
from src.utils.config import ROOT


@pytest.fixture(scope="module")
def api():
    return ScorecardAPI.cargar(modelo="logreg", ruta_modelo="models/pd_logreg.joblib")


@pytest.fixture(scope="module")
def datos():
    return load_splits("pd", "temporal")


@pytest.fixture(scope="module")
def crudos_test():
    """Datos sin preprocesar de la prueba temporal (issue_year == 2015, como en split.py)."""
    df = pd.read_parquet(ROOT / "data" / "processed" / "loans_clean_pd.parquet")
    return df[df["issue_year"] == 2015].reset_index(drop=True)


def test_score_coincide_con_build_scorecard(api, datos):
    modelo, _ = cargar_logistica(ROOT / "models" / "pd_logreg.joblib")
    puntaje, pd_ = api.score(datos["X_test"])
    esperado = Scorecard(600, 50, 20).puntaje_desde_logit(modelo.decision_function(datos["X_test"]))
    assert np.allclose(puntaje, esperado)
    assert np.allclose(pd_, modelo.predict_proba(datos["X_test"])[:, 1])


def test_datos_crudos_reproducen_el_split(api, datos, crudos_test):
    X = api.preprocesar(crudos_test)
    assert X.shape == datos["X_test"].shape
    assert np.allclose(X, datos["X_test"], atol=1e-6)


def test_evaluar_un_solicitante(api, datos, crudos_test):
    r = api.evaluar(crudos_test.iloc[0].to_dict())
    puntaje, _ = api.score(datos["X_test"][:1])
    assert r["puntaje"] == pytest.approx(float(puntaje[0]), abs=1e-3)
    assert 0 <= r["pd"] <= 1 and 0 <= r["percentil"] <= 100


def test_percentil_monotono_y_mediana(api):
    ref = api._referencia
    assert api.percentile(np.array([np.median(ref)]))[0] == pytest.approx(50, abs=0.5)
    assert np.all(np.diff(api.percentile(np.linspace(ref.min(), ref.max(), 50))) >= 0)


def test_percentil_no_depende_de_pdo_ni_base(api, datos):
    otra = ScorecardAPI.cargar(modelo="logreg", ruta_modelo="models/pd_logreg.joblib", pdo=40, base_score=700)
    X = datos["X_test"][:2000]
    assert np.allclose(api.percentile(api.score(X)[0]), otra.percentile(otra.score(X)[0]), atol=0.01)


def test_que_pasaria_si(api, crudos_test):
    fila = crudos_test.iloc[0].to_dict()
    r = api.que_pasaria_si(fila, {"annual_inc": fila["annual_inc"] * 2})
    assert r["cambio_puntaje"] > 0 and r["cambio_pd"] < 0      # beta de annual_inc es negativo
    assert api.que_pasaria_si(fila, {})["cambio_puntaje"] == 0


def test_falta_variable_requerida(api, crudos_test):
    fila = crudos_test.iloc[0].to_dict()
    fila.pop("annual_inc")
    with pytest.raises(ValueError, match="annual_inc"):
        api.evaluar(fila)


def test_categorias(api):
    cats = api.categorias()
    assert "term" in cats and len(cats["term"]) >= 2
    assert set(cats) <= set(api.variables_requeridas)


def test_api_con_la_red_de_p2(datos):
    """La configuración por defecto usa la red final de P2; reproduce su AUC en test temporal."""
    from src.evaluation.metrics import auc
    api_red = ScorecardAPI.cargar()
    puntaje, pd_ = api_red.score(datos["X_test"])
    assert auc(np.asarray(datos["y_test"]), pd_) == pytest.approx(0.6834, abs=1e-3)
    assert np.all(np.isfinite(puntaje))
