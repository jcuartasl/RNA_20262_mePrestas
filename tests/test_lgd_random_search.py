"""Pruebas de src/search/lgd_random_search.py con datos sintéticos y pocas épocas."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.lgd_mlp import entrenar_lgd_mlp, predecir_lgd
from src.search.lgd_random_search import (PERDIDAS, buscar_lgd, calibracion_por_decil, comparar_con_observada,
                                          config_de_fila, configurar, escribir_config_final, mejor_por_perdida,
                                          muestrear_configuraciones, resumir_semillas, semillas_de_config,
                                          techo_tanh)
from src.utils.config import load_config
from tests.test_lgd_mlp import datos_sinteticos

BASE = {"seed": 0, "data": {"scheme": "random"}, "train": {"epochs": 3, "batch_size": 64, "patience": 3},
        "model": {"hidden": [64, 32], "dropout": 0.2, "activation": "relu"},
        "loss": {"name": "mse", "eps": 1e-4}}


def datos():
    X_train, y_train, X_val, y_val = datos_sinteticos()
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}


def test_muestreo_reproducible_y_sin_repetir():
    a, b = muestrear_configuraciones(12, 90), muestrear_configuraciones(12, 90)
    assert a == b and len({str(c) for c in a}) == 12
    assert a != muestrear_configuraciones(12, 91)


def test_configurar_no_modifica_la_base_y_respeta_el_esquema_anidado():
    combinacion = {"hidden": [32], "activation": "tanh", "dropout": 0.1, "lr": 3e-3, "l2": 1e-4}
    cfg = configurar(BASE, combinacion, "beta")
    assert cfg["model"]["hidden"] == [32] and cfg["train"]["lr"] == 3e-3 and cfg["loss"]["name"] == "beta"
    assert cfg["loss"]["eps"] == 1e-4 and cfg["train"]["epochs"] == 3
    assert BASE["model"]["hidden"] == [64, 32] and BASE["loss"]["name"] == "mse"


def test_busqueda_pareada_registra_y_ordena(tmp_path):
    log = tmp_path / "log.csv"
    tabla = buscar_lgd(datos(), BASE, n_configs=2, seed=0, ruta_tabla=tmp_path / "busqueda.csv", ruta_log=log)
    assert len(tabla) == 6 and len(pd.read_csv(log)) == 6 and (tmp_path / "busqueda.csv").exists()
    # cada combinación aparece exactamente una vez con cada pérdida
    assert all(sorted(grupo) == sorted(PERDIDAS) for _, grupo in tabla.groupby("config")["perdida"])
    assert tabla["rmse"].is_monotonic_increasing
    # la beta tiene un parámetro más (phi) que la mse y la bce con la misma arquitectura
    por_config = tabla.pivot(index="config", columns="perdida", values="n_params")
    assert ((por_config["beta"] - por_config["mse"]) == 1).all()
    assert (por_config["bce"] == por_config["mse"]).all()
    mejores = mejor_por_perdida(tabla)
    assert sorted(mejores["perdida"]) == sorted(PERDIDAS) and mejores["rmse"].iloc[0] <= tabla["rmse"].min() + 2.5e-4


def test_desempate_por_dispersion_dentro_de_la_tolerancia():
    tabla = pd.DataFrame({"perdida": ["mse", "bce", "beta"], "rmse": [0.1000, 0.1001, 0.1050],
                          "sd_pred": [0.090, 0.095, 0.120]})
    # mse y bce empatan: gana la menos encogida (bce); la beta queda fuera aunque tenga más dispersión
    assert list(mejor_por_perdida(tabla)["perdida"]) == ["bce", "mse", "beta"]
    # fuera de la tolerancia manda el rmse
    assert list(mejor_por_perdida(tabla, tolerancia=1e-5)["perdida"]) == ["mse", "bce", "beta"]


def test_la_config_final_escrita_reproduce_la_ganadora(tmp_path):
    tabla = buscar_lgd(datos(), BASE, n_configs=1, seed=0, ruta_log=tmp_path / "log.csv")
    ganadora = tabla.iloc[0]
    escribir_config_final(BASE, ganadora, tmp_path / "lgd_mlp.yaml")
    leida = load_config(str(tmp_path / "lgd_mlp.yaml"))
    esperada = config_de_fila(BASE, ganadora)
    for clave in ("data", "train", "model", "loss"): assert leida[clave] == esperada[clave]


def test_semillas_de_config_y_resumen(tmp_path):
    cfg = configurar(BASE, {"hidden": [16], "activation": "relu", "dropout": 0.1, "lr": 1e-2, "l2": 0.0}, "bce")
    tabla, predicciones = semillas_de_config(datos(), cfg, [0, 1, 2], ruta_log=tmp_path / "log.csv")
    assert list(tabla["semilla"]) == [0, 1, 2] and list(predicciones.columns) == ["seed0", "seed1", "seed2"]
    assert len(predicciones) == 200 and len(pd.read_csv(tmp_path / "log.csv")) == 3
    # con dropout, cada semilla da un resultado distinto
    assert tabla["rmse"].nunique() == 3
    resumen = resumir_semillas(tabla)
    assert resumen["rmse_mejor"] == tabla["rmse"].min() and resumen["rmse_peor"] == tabla["rmse"].max()
    assert resumen["r2_mejor"] == tabla["r2"].max() and resumen["n_semillas"] == 3
    assert abs(resumen["rmse_sd"] - tabla["rmse"].std()) < 1e-12


def test_techo_tanh_acota_las_predicciones():
    # con tanh en la última capa oculta ninguna predicción puede superar sigmoide(b + suma |w|)
    d = datos()
    cfg = configurar(BASE, {"hidden": [16], "activation": "tanh", "dropout": 0.0, "lr": 1e-2, "l2": 0.0}, "bce")
    modelo, _, _ = entrenar_lgd_mlp(d["X_train"], d["y_train"], d["X_val"], d["y_val"], cfg)
    salida = modelo.network[-1]
    with torch.no_grad(): salida.weight.fill_(0.1), salida.bias.fill_(-0.2)
    # valor esperado calculado a mano: sigmoide(-0.2 + 16 * 0.1), lejos de 1 para que la cota no sea trivial
    assert techo_tanh(modelo) == pytest.approx(1 / (1 + np.exp(-1.4)), abs=1e-6)
    X_extremo = np.vstack([d["X_val"], 100 * d["X_val"]])     # entradas absurdas para empujar al techo
    assert predecir_lgd(modelo, X_extremo).max() <= techo_tanh(modelo) + 1e-6


def test_calibracion_por_decil():
    p = np.linspace(0, 1, 100)
    tabla = calibracion_por_decil(1 - p, p)                   # y distinto de p: obliga a usar y
    assert len(tabla) == 10 and (tabla["n"] == 10).all()
    np.testing.assert_allclose(tabla["media_obs"], 1 - tabla["media_pred"])
    assert tabla["media_obs"].iloc[0] == pytest.approx(1 - 4.5 / 99)


def test_semillas_de_config_usa_el_nombre_de_experimento_dado(tmp_path):
    cfg = configurar(BASE, {"hidden": [8], "activation": "relu", "dropout": 0.0, "lr": 1e-2, "l2": 0.0}, "bce")
    semillas_de_config(datos(), cfg, [0], ruta_log=tmp_path / "log.csv", experimento="lgd_mlp_relu")
    assert list(pd.read_csv(tmp_path / "log.csv")["experiment"]) == ["lgd_mlp_relu"]


def test_comparar_con_observada_promedia_las_semillas():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, 500)
    semillas = pd.DataFrame({"seed0": 0.5 + 0.1 * (y - 0.5), "seed1": 0.5 + 0.2 * (y - 0.5)})
    resumen, cuantiles = comparar_con_observada(y, {"red": semillas, "lineal": semillas[["seed0"]]})
    red, lineal = resumen.set_index("modelo").loc["red"], resumen.set_index("modelo").loc["lineal"]
    # sd predicha / sd observada vale 0.1 y 0.2 por construcción: la media entre semillas es 0.15
    assert red["n_semillas"] == 2 and red["cociente_sd_media"] == pytest.approx(0.15)
    assert lineal["n_semillas"] == 1 and np.isnan(lineal["cociente_sd_sd"])
    np.testing.assert_allclose(cuantiles["observada"], np.quantile(y, cuantiles.index))
    np.testing.assert_allclose(cuantiles["lineal"], np.quantile(semillas["seed0"], cuantiles.index))
