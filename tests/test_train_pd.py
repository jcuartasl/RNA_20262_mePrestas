"""La demo de PD no debe pisar el modelo real. Copiar a tests/."""
import numpy as np
import pytest

from src.experiments import train_pd


class _Param:
    def numel(self):
        return 10
    requires_grad = True


class _Modelo:
    def parameters(self):
        return [_Param(), _Param()]


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Sustituye datos, entrenamiento y guardado por versiones falsas y redirige models/ a tmp."""
    rng = np.random.default_rng(0)
    datos = {"X_train": rng.normal(size=(1000, 5)).astype(np.float32),
             "y_train": rng.integers(0, 2, 1000).astype(np.float32),
             "X_val": rng.normal(size=(200, 5)).astype(np.float32),
             "y_val": rng.integers(0, 2, 200).astype(np.float32)}
    vistos = {}

    def train_falso(Xt, yt, Xv, yv, cfg):
        vistos["n_train"], vistos["seed"] = len(yt), cfg["seed"]
        return _Modelo(), {"best_val_loss": 0.5, "best_epoch": 1, "epochs_trained": 2}

    def guardar_falso(modelo, cfg, ruta):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text("modelo")

    monkeypatch.setattr(train_pd, "ROOT", tmp_path)
    monkeypatch.setattr(train_pd, "load_splits", lambda *a, **k: datos)
    monkeypatch.setattr(train_pd, "train_pd_mlp", train_falso)
    monkeypatch.setattr(train_pd, "predict_pd", lambda m, X: np.random.default_rng(1).random(len(X)))
    monkeypatch.setattr(train_pd, "save_pd_model", guardar_falso)
    return tmp_path, vistos


def test_entrenamiento_real_guarda_con_el_nombre_oficial(entorno):
    raiz, vistos = entorno
    out = train_pd.train({"seed": 90}, 90)
    assert (raiz / "models" / "pd_mlp_temporal_seed90.pt").exists()
    assert vistos["n_train"] == 1000
    assert {"auc", "gini", "ks", "brier", "bce"} <= set(out["metrics"])


def test_la_demo_no_pisa_el_modelo_real(entorno):
    raiz, vistos = entorno
    real = raiz / "models" / "pd_mlp_temporal_seed90.pt"
    real.parent.mkdir(parents=True)
    real.write_text("MODELO REAL")

    train_pd.train({"seed": 90, "sample_fraction": 0.1}, 90)

    assert real.read_text() == "MODELO REAL"                                  # intacto
    assert (raiz / "models" / "pd_mlp_demo_temporal_seed90.pt").exists()      # la demo va aparte
    assert vistos["n_train"] == 100                                           # y usa la muestra


def test_la_muestra_es_reproducible():
    X, y = np.arange(100).reshape(50, 2), np.arange(50)
    a = train_pd._muestra(X, y, 0.2, 90)[1]
    b = train_pd._muestra(X, y, 0.2, 90)[1]
    assert list(a) == list(b) and len(a) == 10


def test_fraccion_invalida(entorno):
    with pytest.raises(ValueError, match="sample_fraction"):
        train_pd.train({"seed": 90, "sample_fraction": 0}, 90)