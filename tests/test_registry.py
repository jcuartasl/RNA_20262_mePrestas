"""Pruebas de src/utils/registry.py: id de configuración, registro y resumen del presupuesto."""
import csv

import numpy as np
import pandas as pd
import pytest

from src.utils.registry import COLUMNS, config_id, log_run, summarize_budget

CFG = {"model": "MLP", "hidden": [64, 32], "lr": 0.01}


def test_config_id_ignora_semillas_y_rutas():
    a = {**CFG, "seed": 1, "run_seeds": [0, 1], "paths": {"raw": "x"}}
    b = {**CFG, "seed": 99, "run_seeds": [5], "paths": {"raw": "y"}}
    assert config_id(a) == config_id(b) == config_id(CFG)


def test_config_id_cambia_con_un_hiperparametro():
    assert config_id(CFG) != config_id({**CFG, "lr": 0.1})
    assert config_id(CFG) != config_id({**CFG, "hidden": [64, 64]})


def test_config_id_no_depende_del_orden_de_las_claves():
    assert config_id({"a": 1, "b": 2}) == config_id({"b": 2, "a": 1})


def test_log_run_crea_encabezado_una_sola_vez(tmp_path):
    log = tmp_path / "sub" / "log.csv"              # la carpeta no existe todavía
    log_run("exp", CFG, 0, 100, 1.0, {"auc": 0.8}, log_path=log)
    log_run("exp", CFG, 1, 100, 1.0, {"auc": 0.9}, log_path=log)

    with open(log, newline="", encoding="utf-8") as f:
        filas = list(csv.reader(f))
    assert filas[0] == COLUMNS
    assert len(filas) == 3                          # encabezado + 2 corridas
    assert filas[1][COLUMNS.index("experiment")] == "exp"


def test_log_run_acepta_tipos_de_numpy_en_metrics(tmp_path):
    log = tmp_path / "log.csv"
    log_run("exp", CFG, 0, 10, 0.5, {"auc": np.float32(0.75), "n": np.int64(3)}, log_path=log)
    assert pd.read_csv(log).shape[0] == 1


def test_resumen_cuenta_configuraciones_y_corridas(tmp_path):
    log = tmp_path / "log.csv"
    for seed in range(3):
        log_run("exp_a", CFG, seed, 1000, 1.5, {"auc": 0.8}, log_path=log)
    log_run("exp_b", {**CFG, "lr": 0.1}, 0, 2000, 2.0, {"auc": 0.7}, log_path=log)

    res = summarize_budget(log_path=log)

    assert res.loc["exp_a", "n_configs"] == 1       # 3 semillas = 1 configuración
    assert res.loc["exp_a", "n_runs"] == 3
    assert res.loc["exp_a", "total_seconds"] == 4.5
    assert res.loc["exp_a", "n_params"] == 1000
    assert res.loc["exp_b", "n_configs"] == 1
    assert res.loc["exp_b", "n_runs"] == 1


def test_resumen_distingue_configuraciones(tmp_path):
    log = tmp_path / "log.csv"
    log_run("exp", CFG, 0, 10, 1.0, {}, log_path=log)
    log_run("exp", {**CFG, "lr": 0.5}, 0, 10, 1.0, {}, log_path=log)
    assert summarize_budget(log_path=log).loc["exp", "n_configs"] == 2


def test_resumen_filtra_por_experimento(tmp_path):
    log = tmp_path / "log.csv"
    log_run("exp_a", CFG, 0, 10, 1.0, {}, log_path=log)
    log_run("exp_b", CFG, 0, 10, 1.0, {}, log_path=log)
    res = summarize_budget("exp_a", log_path=log)
    assert list(res.index) == ["exp_a"]


def test_resumen_sin_registro_da_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        summarize_budget(log_path=tmp_path / "no_existe.csv")