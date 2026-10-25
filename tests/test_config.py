"""Pruebas de src/utils/config.py: parseo de overrides, herencia del default y mezcla."""
import pytest
import yaml

from src.utils.config import DEFAULT_PATH, _merge, _parse, load_config


@pytest.mark.parametrize("raw, esperado", [
    ("0.001", 0.001), ("1e-3", 0.001), ("64", 64),
    ("[64, 32]", [64, 32]), ("relu", "relu"),
    ("true", True), ("False", False),
    ("null", None), ("None", None), ("none", None),
    ("  0.5 ", 0.5),
])
def test_parse(raw, esperado):
    assert _parse(raw) == esperado
    # Evita que 1 se confunda con True, o 0.0 con None
    assert type(_parse(raw)) is type(esperado)


def test_merge_recursivo_no_muta_la_base():
    base = {"a": 1, "model": {"hidden": [64, 32], "dropout": 0.2}}
    nuevo = {"model": {"dropout": 0.5}, "b": 2}
    out = _merge(base, nuevo)
    assert out == {"a": 1, "b": 2, "model": {"hidden": [64, 32], "dropout": 0.5}}
    assert base["model"]["dropout"] == 0.2         


def test_hereda_default_desde_cualquier_directorio(tmp_path, monkeypatch):
    """Regresión del bug 1: DEFAULT_PATH no debe depender del directorio de trabajo."""
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"model": {"hidden": [8]}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)                     # cwd distinto de la raíz del repo

    cfg = load_config(str(exp))

    assert cfg["model"]["hidden"] == [8]            # lo del experimento
    assert "seed" in cfg                            # lo heredado de default.yaml
    assert len(cfg["run_seeds"]) == 30


def test_yaml_del_experimento_pisa_al_default(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"seed": 123}), encoding="utf-8")
    assert load_config(str(exp))["seed"] == 123


def test_cargar_el_propio_default_no_falla():
    cfg = load_config(str(DEFAULT_PATH))
    assert cfg["seed"] == 90


def test_yaml_vacio_devuelve_solo_el_default(tmp_path):
    exp = tmp_path / "vacio.yaml"
    exp.write_text("", encoding="utf-8")
    assert load_config(str(exp)) == load_config(str(DEFAULT_PATH))


def test_override_anidado_y_tipos(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"train": {"lr": 0.001, "epochs": 100}}), encoding="utf-8")

    cfg = load_config(str(exp), ["train.lr=0.01", "model.hidden=[128, 64]", "model.dropout=null"])

    assert cfg["train"]["lr"] == 0.01               # lo sobrescrito
    assert cfg["train"]["epochs"] == 100            # lo demás queda como estaba
    assert cfg["model"]["hidden"] == [128, 64]
    assert cfg["model"]["dropout"] is None


def test_override_crea_seccion_nueva(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("{}", encoding="utf-8")
    cfg = load_config(str(exp), ["extra.nivel.valor=3"])
    assert cfg["extra"]["nivel"]["valor"] == 3


def test_override_sobre_un_valor_que_no_es_seccion_da_error_claro(tmp_path):
    """train es un número aquí: 'train.lr' debe fallar con ValueError explicativo."""
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"train": 0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="train"):
        load_config(str(exp), ["train.lr=0.001"])