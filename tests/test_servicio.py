"""Coherencia entre el formulario/API y los artefactos guardados (preprocessor.pkl, scorecard, LGD)."""
import numpy as np
import pytest

pytest.importorskip("torch")
from app.servicio import FIJAS, SIN_EFECTO, obtener_servicio  # noqa: E402


@pytest.fixture(scope="module")
def s():
    return obtener_servicio()


def test_formulario_cubre_el_preprocesador(s):
    """Toda variable que exige el modelo se pregunta o se fija explícitamente: nada queda sin documentar."""
    requeridas = set(s.sc.variables_requeridas)
    preguntadas = {c["clave"] for c in s.campos}
    cubiertas = preguntadas | {"funded_amnt", "funded_amnt_inv", "initial_list_status"} | set(FIJAS) | set(SIN_EFECTO)
    assert requeridas <= cubiertas, requeridas - cubiertas
    assert preguntadas <= requeridas  # no se pregunta nada que el modelo ignore


def test_categorias_del_formulario_son_las_del_modelo(s):
    cats = s.sc.categorias()
    for c in s.campos:
        if c["tipo"] == "categoria":
            assert [o["valor"] for o in c["opciones"]] == cats[c["clave"]]


def test_funded_amnt_igual_al_monto(s):
    fila, _, _ = s.completar({"loan_amnt": 20000})
    assert fila["funded_amnt"] == fila["funded_amnt_inv"] == 20000


def test_earliest_cr_line_no_cambia_el_resultado(s):
    """Hallazgo: todas las categorías mes-año quedaron 'infrecuentes' en el preprocesador."""
    fila, _, _ = s.completar({})
    a = s.sc.preprocesar({**fila, "earliest_cr_line": "Jan-1970"})
    b = s.sc.preprocesar({**fila, "earliest_cr_line": "Apr-1955"})
    assert np.allclose(a, b)


def test_pd_puntaje_consistentes(s):
    r = s.calcular({"dti": 20})
    assert 0 < r["PD"] < 1 and 0 <= r["LGD"] <= 1 and 0 <= r["Percentil"] <= 100
    assert r["Perdida_esperada_USD"] == pytest.approx(r["PD"] * r["LGD"] * r["EAD_USD"])
    sc = s.sc.scorecard
    assert sc.pd_desde_puntaje(r["Puntaje"]) == pytest.approx(r["PD"], rel=1e-4)


def test_mas_riesgo_menos_puntaje(s):
    assert s.calcular({"term": " 60 months", "revol_util": 95})["Puntaje"] < s.calcular({"term": " 36 months", "revol_util": 20})["Puntaje"]


@pytest.mark.parametrize("malo", [{"foo": 1}, {"term": " 48 months"}, {"dti": -1}, {"annual_inc": "abc"}, {"pub_rec": 1.5}])
def test_entradas_invalidas(s, malo):
    with pytest.raises(ValueError):
        s.calcular(malo)


def test_extrapolacion_genera_aviso(s):
    assert any("extrapolación" in a for a in s.calcular({"annual_inc": 5_000_000})["avisos"])
