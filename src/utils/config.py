"""Lectura de configuración YAML en default.yaml y modificación de valores por línea de comando."""
import ast
import copy
import os
import yaml

DEFAULT_PATH = "../../configs/default.yaml"


def _parse(value: str):
    """Convierte '0.001' -> float, '[64,32]' -> lista, 'relu' -> str."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def _merge(base: dict, new: dict) -> dict:
    """Mezcla recursiva: los valores de 'new' reemplazan a los de 'base'."""
    out = copy.deepcopy(base)
    for key, vals in new.items():
        if isinstance(vals, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], vals)
        else:
            out[key] = vals
    return out


def _read(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(path: str, overrides=None, default_path: str = DEFAULT_PATH) -> dict:
    """Carga default.yaml, le superpone `path` y aplica overrides tipo 'train.lr=0.001'."""
    cfg = {}
    if os.path.exists(default_path) and os.path.abspath(path) != os.path.abspath(default_path):
        cfg = _read(default_path)
    cfg = _merge(cfg, _read(path))

    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[last] = _parse(value)
    return cfg