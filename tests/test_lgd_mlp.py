"""Pruebas de src/models/lgd_mlp.py y del contrato de src/experiments/train_lgd.py con datos sintéticos."""
import numpy as np
import pytest
import torch

import src.experiments.train_lgd as train_lgd
from src.models.lgd_mlp import (_opciones, ajustar_beta_marginal, cargar_lgd_mlp, contar_parametros,
                                entrenar_lgd_mlp, figuras_distribucion_lgd, guardar_lgd_mlp, metricas_lgd,
                                predecir_lgd)


def datos_sinteticos(n=600, seed=0):
    # lgd simulada de una beta cuya media depende de las variables, como supone la red
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    mu = 1 / (1 + np.exp(-(X @ np.array([1.0, -0.8, 0.5, 0.0, 0.3]) + 0.8)))
    y = rng.beta(mu * 6, (1 - mu) * 6).astype(np.float32)
    return X[:400], y[:400], X[400:], y[400:]


def cfg_rapida(perdida="mse", **train):
    return {"model": {"hidden": [16], "dropout": 0.0}, "loss": {"name": perdida},
            "train": {"lr": 0.01, "epochs": 60, "batch_size": 64, "patience": 10, **train}}


def test_opciones_traduce_la_config_anidada_y_el_atajo():
    op = _opciones({"model": {"hidden": [8, 4]}, "train": {"lr": 0.05}, "loss": "beta"})
    assert op["hidden_layers"] == [8, 4] and op["lr"] == 0.05 and op["perdida"] == "beta"
    # con la config de la demo (sin model ni loss) se usan los valores por defecto
    assert _opciones({"train": {"epochs": 3}})["perdida"] == "mse"


def test_sobreajusta_un_lote_pequeno():
    # si el bucle está bien, 64 ejemplos se memorizan: la mse de entrenamiento baja casi a 0
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(64, 5)).astype(np.float32), rng.uniform(0.05, 0.95, 64).astype(np.float32)
    cfg = {"model": {"hidden": [64], "dropout": 0.0}, "train": {"lr": 0.01, "epochs": 1500, "batch_size": 64,
                                                                 "patience": 1500}}
    _, _, historial = entrenar_lgd_mlp(X, y, X, y, cfg)
    assert historial["best_val_loss"] < 1e-4


def test_aprende_mejor_que_predecir_la_media():
    X_train, y_train, X_val, y_val = datos_sinteticos()
    modelo, _, _ = entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg_rapida())
    rmse_media = np.sqrt(np.mean((y_val - y_train.mean()) ** 2))
    assert metricas_lgd(y_val, predecir_lgd(modelo, X_val))["rmse"] < 0.9 * rmse_media


@pytest.mark.parametrize("perdida", ["mse", "bce", "beta"])
def test_las_tres_perdidas_entrenan(perdida):
    X_train, y_train, X_val, y_val = datos_sinteticos()
    modelo, log_phi, historial = entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg_rapida(perdida))
    p = predecir_lgd(modelo, X_val)
    assert np.isfinite(historial["val_loss"]).all() and ((p >= 0) & (p <= 1)).all()
    assert (log_phi is not None) == (perdida == "beta")
    # phi arranca en 4 y debe moverse hacia la precisión de los datos (se simularon con phi = 6)
    if perdida == "beta": assert log_phi.item() != pytest.approx(np.log(4.0), abs=1e-3)


def test_misma_semilla_mismo_resultado_y_otra_semilla_cambia():
    X_train, y_train, X_val, y_val = datos_sinteticos()
    cfg = cfg_rapida(epochs=10, patience=10) | {"model": {"hidden": [16], "dropout": 0.2}}
    a, b, c = (predecir_lgd(entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg, s)[0], X_val)
               for s in (0, 0, 1))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_parada_temprana_restaura_la_mejor_epoca():
    X_train, y_train, X_val, y_val = datos_sinteticos()
    cfg = cfg_rapida(lr=0.05, epochs=300, patience=5) | {"model": {"hidden": [128, 128], "dropout": 0.0}}
    modelo, _, historial = entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg)
    assert historial["epochs_trained"] < 300
    assert historial["best_val_loss"] == min(historial["val_loss"])
    assert historial["val_loss"][historial["best_epoch"] - 1] == historial["best_val_loss"]
    # el modelo devuelto es el de la mejor época, no el de la última
    mse_val = np.mean((y_val - predecir_lgd(modelo, X_val)) ** 2)
    assert mse_val == pytest.approx(historial["best_val_loss"], rel=1e-5)


def test_guardar_y_cargar_da_las_mismas_predicciones(tmp_path):
    X_train, y_train, X_val, y_val = datos_sinteticos()
    cfg = cfg_rapida("beta", epochs=5)
    modelo, log_phi, _ = entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg)
    guardar_lgd_mlp(modelo, log_phi, cfg, tmp_path / "lgd.pt")
    cargado, log_phi_cargado, perdida = cargar_lgd_mlp(tmp_path / "lgd.pt")
    np.testing.assert_array_equal(predecir_lgd(cargado, X_val), predecir_lgd(modelo, X_val))
    assert log_phi_cargado == pytest.approx(log_phi.item()) and perdida == "beta"


@pytest.fixture
def contrato(tmp_path, monkeypatch):
    """train_lgd.train con datos sintéticos y escribiendo los modelos en una carpeta temporal."""
    X_train, y_train, X_val, y_val = datos_sinteticos()
    datos = {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
    monkeypatch.setattr(train_lgd, "load_splits", lambda task, scheme: datos)
    monkeypatch.setattr(train_lgd, "ROOT", tmp_path)
    (tmp_path / "models").mkdir()
    return tmp_path


def test_contrato_de_train(contrato):
    salida = train_lgd.train(cfg_rapida(), seed=3)
    assert set(salida["metrics"]) == {"rmse", "mae", "r2"} and isinstance(salida["n_params"], int)
    assert (contrato / "models" / "lgd_mlp_seed3.pt").exists()
    # 5 -> 16 -> 1: 5*16 + 16 + 16 + 1 = 113 parámetros; con beta se suma phi
    assert salida["n_params"] == 113
    assert train_lgd.train(cfg_rapida() | {"loss": "beta"}, seed=3)["n_params"] == 114


def test_contrato_con_la_config_de_la_demo(contrato):
    # demo.py solo pasa default.yaml + demo.yaml: train.epochs y sample_fraction, sin model ni loss
    salida = train_lgd.train({"seed": 90, "train": {"epochs": 3}, "sample_fraction": 0.1}, seed=90)
    assert np.isfinite(list(salida["metrics"].values())).all()
    # la demo no debe pisar el modelo entrenado con todos los datos
    assert (contrato / "models" / "lgd_mlp_demo_seed90.pt").exists()
    assert not (contrato / "models" / "lgd_mlp_seed90.pt").exists()
    # y debe entrenar la red final de configs/lgd_mlp.yaml, no la de los valores por defecto (mse, relu)
    guardado = torch.load(contrato / "models" / "lgd_mlp_demo_seed90.pt", weights_only=True)
    assert guardado["perdida"] == "bce" and guardado["architecture"]["activation"] == "tanh"
    assert guardado["architecture"]["hidden_layers"] == [128, 64, 32]


def test_un_override_si_cambia_el_entrenamiento(contrato):
    # evita el error silencioso de pasar la config anidada a funciones que esperan claves planas
    base = train_lgd.train(cfg_rapida(epochs=5), seed=0)["metrics"]
    otra = train_lgd.train(cfg_rapida(epochs=5, lr=0.0001), seed=0)["metrics"]
    assert base["rmse"] != otra["rmse"]


def test_ajustar_beta_marginal_recupera_los_parametros():
    y = np.random.default_rng(0).beta(0.7 * 4, 0.3 * 4, 20000)
    mu, phi = ajustar_beta_marginal(y)
    assert mu == pytest.approx(0.7, abs=0.01) and phi == pytest.approx(4, rel=0.05)


def test_contar_parametros():
    X_train, y_train, X_val, y_val = datos_sinteticos()
    modelo, _, _ = entrenar_lgd_mlp(X_train, y_train, X_val, y_val, cfg_rapida(epochs=1))
    assert contar_parametros(modelo) == 113 and contar_parametros(modelo, log_phi=object()) == 114


def test_figuras_crea_la_carpeta_si_no_existe(tmp_path):
    carpeta = tmp_path / "no" / "existe"
    figuras_distribucion_lgd(carpeta)
    nombres = ["lgd_ajuste_beta_normal.png", "lgd_qq_beta_normal.png", "lgd_varianza_por_decil.png"]
    assert all((carpeta / nombre).exists() for nombre in nombres)
