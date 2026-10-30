"""Pruebas de src/models/pd_logreg.py con datos sintéticos (rápidas, no tocan los datos reales)."""
import inspect
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.models.losses import get_loss, penalizacion_l2
from src.models.mlp import build_mlp
from src.models.pd_logreg import (ajustar_platt, aplicar_platt, buscar_regularizacion, cargar_logistica,
                                  entrenar_logistica, guardar_logistica, predecir_pd_logistica, recalibrar)
from src.utils.config import ROOT


def datos_sinteticos():
    # sin variables redundantes para que la logística sin penalizar tenga solución única
    X, y = make_classification(n_samples=2000, n_features=8, n_informative=4, n_redundant=0, random_state=0)
    return X[:1500], y[:1500].astype(float), X[1500:], y[1500:].astype(float)


def ajustar_en_torch(X, y, lam):
    # logística = mlp sin capas ocultas; se minimiza bce + lam * 0.5 |w|^2 con lbfgs hasta converger
    modelo = build_mlp({"input_dim": X.shape[1], "hidden_layers": [], "dropout": 0.0}).double()
    X, y = torch.from_numpy(X), torch.from_numpy(y)
    optimizador = torch.optim.LBFGS(modelo.parameters(), max_iter=500, tolerance_grad=1e-12,
                                    tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def paso():
        optimizador.zero_grad()
        perdida = get_loss("bce")(modelo(X).squeeze(1), y) + lam * penalizacion_l2(modelo)
        perdida.backward()
        return perdida

    optimizador.step(paso)
    capa = modelo.network[0]
    return capa.weight.detach().numpy().ravel(), capa.bias.item()


def test_bce_propia_da_la_misma_logistica_que_sklearn():
    # minimizar la bce de losses.py es máxima verosimilitud: debe dar la logística sin penalizar de sklearn
    # tol estricta porque con la de sklearn por defecto (1e-4) se detiene a ~3e-4 del óptimo
    X, y, _, _ = datos_sinteticos()
    sk = entrenar_logistica(X, y, C=np.inf, l1_ratio=0.0, tol=1e-10)
    w, b = ajustar_en_torch(X, y, lam=0.0)
    np.testing.assert_allclose(w, sk.coef_.ravel(), atol=1e-6)
    assert b == pytest.approx(sk.intercept_[0], abs=1e-6)


def test_relacion_entre_C_y_lambda():
    # sklearn minimiza C * suma(nll) + 0.5 |w|^2, que es lo mismo que media(nll) + (1 / (n C)) * 0.5 |w|^2
    X, y, _, _ = datos_sinteticos()
    C = 0.01
    sk = entrenar_logistica(X, y, C=C, l1_ratio=0.0, max_iter=5000, tol=1e-10)
    w, b = ajustar_en_torch(X, y, lam=1 / (len(y) * C))
    np.testing.assert_allclose(w, sk.coef_.ravel(), atol=1e-6)
    assert b == pytest.approx(sk.intercept_[0], abs=1e-6)


def test_l1_deja_ceros_exactos_y_mas_con_C_pequeno():
    X, y, _, _ = datos_sinteticos()
    ceros = [int((entrenar_logistica(X, y, C, l1_ratio=1.0).coef_ == 0).sum()) for C in [1.0, 0.01, 0.002]]
    assert ceros == sorted(ceros) and ceros[-1] > 0
    # con l2 los coeficientes se encogen pero ninguno llega a 0 exacto
    assert (entrenar_logistica(X, y, 0.002, l1_ratio=0.0).coef_ != 0).all()


def test_busqueda_registra_y_elige_la_menor_bce(tmp_path):
    X_train, y_train, X_val, y_val = datos_sinteticos()
    log, ruta_tabla = tmp_path / "log.csv", tmp_path / "busqueda.csv"
    modelo, tabla = buscar_regularizacion(X_train, y_train, X_val, y_val, [0.01, 1.0], [0.0, 1.0],
                                          ruta_tabla=ruta_tabla, ruta_log=log)
    assert len(tabla) == 5 and len(pd.read_csv(log)) == 5 and ruta_tabla.exists()
    assert tabla["bce"].is_monotonic_increasing and (tabla["n_params"] == 9).all()
    p = predecir_pd_logistica(modelo, X_val)
    bce = -np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))
    assert bce == pytest.approx(tabla["bce"][0])
    # la búsqueda solo recibe train y validación: el test no se usa para elegir
    assert not any("test" in nombre for nombre in inspect.signature(buscar_regularizacion).parameters)


def test_guardar_y_cargar_da_las_mismas_predicciones(tmp_path):
    X, y, X_val, _ = datos_sinteticos()
    modelo = entrenar_logistica(X, y, C=1.0, l1_ratio=0.0)
    guardar_logistica(modelo, [f"x{i}" for i in range(8)], tmp_path / "logistica.joblib")
    cargado, variables = cargar_logistica(tmp_path / "logistica.joblib")
    np.testing.assert_array_equal(predecir_pd_logistica(cargado, X_val), predecir_pd_logistica(modelo, X_val))
    assert variables == [f"x{i}" for i in range(8)]


def test_platt_corrige_el_desplazamiento_y_no_cambia_la_auc():
    # cambio de época simulado: el logit real está 0.6 por encima del que predice el modelo
    rng = np.random.default_rng(0)
    z = rng.normal(-1.5, 1.0, 20000)
    p, y = expit(z), (rng.random(20000) < expit(z + 0.6)).astype(float)
    a, b = ajustar_platt(p, y)
    q = aplicar_platt(p, a, b)
    assert a == pytest.approx(0.6, abs=0.1) and b == pytest.approx(1.0, abs=0.1)
    # con intercepto sin penalizar, la pd media recalibrada iguala la tasa (como en la sección 5.1)
    assert q.mean() == pytest.approx(y.mean(), abs=1e-3) and p.mean() < y.mean() - 0.05
    # b > 0 es una transformación creciente: no cambia el orden ni la auc
    assert roc_auc_score(y, q) == pytest.approx(roc_auc_score(y, p), abs=1e-12)


def test_recalibrar_solo_usa_validacion(tmp_path):
    X_train, y_train, X_val, y_val = datos_sinteticos()
    modelo = entrenar_logistica(X_train, y_train, C=1.0, l1_ratio=0.0)
    datos = {"X_val": X_val, "y_val": y_val, "X_test": X_val[:200], "y_test": y_val[:200]}
    tabla, a, b = recalibrar(modelo, datos, tmp_path / "platt.csv")
    assert len(tabla) == 4 and (tmp_path / "platt.csv").exists()
    # cambiar las etiquetas de prueba no puede mover a ni b
    datos["y_test"] = 1 - datos["y_test"]
    assert recalibrar(modelo, datos)[1:] == (a, b)


def test_comando_arranca():
    r = subprocess.run([sys.executable, "-m", "src.models.pd_logreg", "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "--override" in r.stdout
