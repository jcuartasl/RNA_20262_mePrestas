"""Pruebas de src/utils/cli.py y de que cada comando del README arranca."""
import subprocess
import sys
from argparse import Namespace

import numpy as np
import pytest

from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT

EXPERIMENTOS = ["prepare_data", "train_pd", "train_lgd", "build_scorecard",
                "expected_loss", "make_figures", "demo"]


def test_parser_valores_por_defecto():
    args = build_parser("x", "configs/pd_mlp.yaml").parse_args([])
    assert args.config == "configs/pd_mlp.yaml"
    assert args.override == []


def test_parser_acepta_varios_overrides():
    args = build_parser("x").parse_args(["--override", "train.lr=0.01", "model.hidden=[8,4]"])
    assert args.override == ["train.lr=0.01", "model.hidden=[8,4]"]


def test_init_run_aplica_overrides_y_fija_la_semilla():
    args = Namespace(config="configs/default.yaml", override=["seed=7", "split.train=0.8"])
    cfg = init_run(args)
    a = np.random.rand()
    init_run(args)
    assert cfg["seed"] == 7 and cfg["split"]["train"] == 0.8
    assert np.random.rand() == a                    # misma semilla -> mismo número


def test_init_run_resuelve_rutas_relativas_desde_la_raiz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                     # ejecutando desde otra carpeta
    cfg = init_run(Namespace(config="configs/default.yaml", override=[]))
    assert cfg["seed"] == 90


@pytest.mark.parametrize("modulo", EXPERIMENTOS)
def test_cada_comando_arranca(modulo):
    """python -m src.experiments.<modulo> debe imprimir su config y salir con código 0."""
    r = subprocess.run([sys.executable, "-m", f"src.experiments.{modulo}"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_override_llega_al_comando():
    r = subprocess.run([sys.executable, "-m", "src.experiments.prepare_data",
                        "--override", "seed=7"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "'seed': 7" in r.stdout


def test_demo_marca_los_pasos_sin_implementar():
    r = subprocess.run([sys.executable, "-m", "src.experiments.demo"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "pendiente (stub)" in r.stdout or "ok" in r.stdout