"""Pruebas de la pérdida esperada (sin torch ni parquet: se simulan la cartera y los modelos)."""
import numpy as np
import pandas as pd
import pytest

from src.expected_loss.el import expected_loss
from src.experiments import expected_loss as xel


def test_el_es_producto_y_usa_float64():
    r = expected_loss([0.1, 0.5], [0.4, 1.0], [1000.0, 200.0])
    assert r["el_individual"].dtype == np.float64
    assert np.allclose(r["el_individual"], [40.0, 100.0])
    assert r["el_total"] == pytest.approx(140.0)
    assert r["el_rate"] == pytest.approx(140.0 / 1200.0)


def test_el_dimensiones_inconsistentes():
    with pytest.raises(ValueError):
        expected_loss([0.1], [0.4, 0.5], [1.0, 2.0])


def test_resumir_perdida_realizada_y_cociente():
    pd_, lgd, ead = np.array([0.2, 0.4, 0.1]), np.array([0.5, 0.5, 0.5]), np.array([100.0, 100.0, 200.0])
    real = np.array([0.0, 50.0, 0.0])
    f = xel.resumir("x", pd_, lgd, ead, real, np.array([0, 1, 0]))
    assert f["el_total"] == pytest.approx(10 + 20 + 10)
    assert f["perdida_real_total"] == 50.0
    assert f["el_sobre_real"] == pytest.approx(0.8)


def test_resumir_naive_constante_no_rompe_spearman():
    f = xel.resumir("naive", np.full(3, 0.3), np.full(3, 0.6), np.full(3, 100.0), np.array([0.0, 60.0, 0.0]),
                    np.array([0, 1, 0]))
    assert np.isnan(f["spearman_el_vs_real"])


def test_ruta_lgd_cae_a_aleatoria_con_aviso(tmp_path, monkeypatch):
    monkeypatch.setattr(xel, "ROOT", tmp_path)
    ruta, aviso = xel.ruta_lgd_mlp({"lgd_scheme": "temporal", "lgd_seed": 3})
    assert ruta == "models/lgd_mlp_seed3.pt" and "AVISO" in aviso
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "lgd_mlp_temporal_seed3.pt").write_bytes(b"x")
    ruta, aviso = xel.ruta_lgd_mlp({"lgd_scheme": "temporal", "lgd_seed": 3})
    assert ruta == "models/lgd_mlp_temporal_seed3.pt" and aviso == ""


def test_cartera_detecta_desalineacion(monkeypatch):
    df_pd = pd.DataFrame({"target_pd": [0, 1, 1, 0], "funded_amnt": [1.0, 2.0, 3.0, 4.0],
                          "issue_year": [2013, 2014, 2015, 2015]})
    df_lgd = pd.DataFrame({"ead": [2.0, 3.0], "lgd": [0.5, 0.8], "issue_year": [2012, 2015]})
    tablas = {"loans_clean_pd.parquet": df_pd, "loans_clean_lgd.parquet": df_lgd}
    monkeypatch.setattr(pd, "read_parquet", lambda ruta: tablas[ruta.name])
    import src.data as datos
    monkeypatch.setattr(datos, "load_splits", lambda t, e: {"X_test": np.zeros((2, 3)), "y_test": np.array([1.0, 0.0])})
    c = xel.cargar_cartera({})
    assert np.allclose(c["perdida_real"], [0.8 * 3.0, 0.0])
    assert c["lgd_media_train"] == pytest.approx(0.5)
    tablas["loans_clean_lgd.parquet"] = df_lgd.iloc[:1]          # un incumplido menos -> error
    with pytest.raises(ValueError):
        xel.cargar_cartera({})
