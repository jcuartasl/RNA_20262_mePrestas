"""Pruebas de src/experiments/run_seeds.py con un entrenamiento falso (no entrena nada real)."""
import sys

import numpy as np
import pandas as pd
import pytest

from src.experiments import run_seeds
from src.utils import registry


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Redirige el log a una carpeta temporal y sustituye train() de PD y LGD por uno falso."""
    log = tmp_path / "log.csv"
    llamadas = []

    def train_falso(cfg, seed):
        llamadas.append((seed, float(np.random.rand())))    # un número que depende de la semilla
        return {"metrics": {"auc": 0.5 + seed / 1000}, "n_params": 1234}

    monkeypatch.setattr("src.experiments.train_pd.train", train_falso)
    monkeypatch.setattr("src.experiments.train_lgd.train", train_falso)
    # run_seeds llama a log_run sin log_path: se le fuerza el temporal
    monkeypatch.setattr(run_seeds, "log_run",
                        lambda *a, **k: registry.log_run(*a, log_path=log, **k))
    return log, llamadas


def _ejecutar(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["run_seeds", *args])
    run_seeds.main()


def test_por_defecto_corre_las_30_semillas(entorno, monkeypatch):
    log, llamadas = entorno
    _ejecutar(monkeypatch, "--model", "pd_mlp")

    df = pd.read_csv(log)
    assert len(df) == 30
    assert list(df["seed"]) == list(range(30))
    assert set(df["experiment"]) == {"pd_mlp"}
    assert df["config_id"].nunique() == 1           # 30 semillas = 1 configuración
    assert registry.summarize_budget(log_path=log).loc["pd_mlp", "n_runs"] == 30


def test_n_seeds_limita_las_corridas(entorno, monkeypatch):
    log, _ = entorno
    _ejecutar(monkeypatch, "--model", "pd_mlp", "--n_seeds", "5")
    assert list(pd.read_csv(log)["seed"]) == [0, 1, 2, 3, 4]


def test_funciona_tambien_con_lgd(entorno, monkeypatch):
    log, _ = entorno
    _ejecutar(monkeypatch, "--model", "lgd_mlp", "--n_seeds", "2")
    assert set(pd.read_csv(log)["experiment"]) == {"lgd_mlp"}


def test_cada_corrida_usa_su_semilla_y_es_reproducible(entorno, monkeypatch):
    log, llamadas = entorno
    _ejecutar(monkeypatch, "--model", "pd_mlp", "--n_seeds", "4")
    primera = list(llamadas)
    llamadas.clear()
    _ejecutar(monkeypatch, "--model", "pd_mlp", "--n_seeds", "4")

    assert [s for s, _ in primera] == [0, 1, 2, 3]
    assert primera == llamadas                      # mismos números al repetir
    assert len({v for _, v in primera}) == 4        # semillas distintas, valores distintos


def test_modelo_desconocido_se_rechaza(entorno, monkeypatch):
    with pytest.raises(SystemExit):
        _ejecutar(monkeypatch, "--model", "no_existe")