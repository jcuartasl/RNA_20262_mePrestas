"""Lectura de configuración YAML (default.yaml + experimento) y overrides por línea de comandos.

Cambios respecto a la versión anterior (punto 2.2 de la auditoría):
  * `load_config(..., strict=True)` rechaza overrides cuya clave NO existe en la config ya cargada
    (default.yaml + YAML del experimento). Antes `--override lr=0.001` creaba una clave suelta que el
    entrenamiento ignoraba en silencio.
  * El error sugiere la clave correcta ("lr" -> "train.lr") y lista las claves válidas cercanas.
  * También se validan: formato CLAVE=VALOR, claves vacías y reemplazar una sección por un escalar.
  * `strict=False` conserva el comportamiento anterior (crea claves nuevas); lo usan los tests y
    cualquier script que realmente necesite añadir secciones.
"""
import ast
import copy
import difflib
import os
from pathlib import Path

import yaml
from src.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "configs" / "default.yaml"

_LITERALS = {
    "true": True,
    "false": False,
    "none": None,
    "null": None,
}

EXTRA_KEYS = {
    "split_seed",       
    "sample_fraction",  
}


def _parse(value: str):
    """Convierte '0.001' -> float, '[64,32]' -> lista, 'relu' -> str."""
    value = value.strip()
    try:
        if value.lower() in _LITERALS:
            return _LITERALS[value.lower()]
        return ast.literal_eval(value)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
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


def claves_validas(cfg: dict, prefijo: str = "") -> set:
    """Todas las rutas con punto de la config: secciones y hojas ('train', 'train.lr', ...)."""
    rutas = set()
    for k, v in cfg.items():
        ruta = f"{prefijo}{k}"
        rutas.add(ruta)
        if isinstance(v, dict):
            rutas |= claves_validas(v, ruta + ".")
    return rutas


def _split_override(item: str):
    """'train.lr=0.001' -> ('train.lr', '0.001'), con errores claros."""
    if "=" not in item:
        raise ValueError(f"Override '{item}' inválido: el formato es CLAVE=VALOR (p. ej. train.lr=0.001).")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key or any(not parte for parte in key.split(".")):
        raise ValueError(f"Override '{item}' inválido: la clave '{key}' está vacía o tiene un punto sobrante.")
    return key, value


def _error_clave_desconocida(item: str, key: str, conocidas: set) -> ValueError:
    ultimo = key.split(".")[-1]
    # 1) misma hoja en otra sección: 'lr' -> 'train.lr'
    misma_hoja = sorted(r for r in conocidas if r.split(".")[-1] == ultimo and r != key)
    # 2) claves parecidas por ortografía: 'train.lrr' -> 'train.lr'
    parecidas = [r for r in difflib.get_close_matches(key, sorted(conocidas), n=3, cutoff=0.6)
                 if r not in misma_hoja]
    sugeridas = (misma_hoja + parecidas)[:4]
    msg = f"Override '{item}': la clave '{key}' no existe en la configuración y sería ignorada."
    if sugeridas:
        msg += " ¿Quisiste decir: " + ", ".join(sugeridas) + "?"
    else:
        hojas = sorted(r for r in conocidas if "." not in r or r.split(".")[0] == key.split(".")[0])
        msg += " Claves disponibles: " + ", ".join(hojas[:15]) + (" ..." if len(hojas) > 15 else "")
    return ValueError(msg)


def validar_override(item: str, cfg_base: dict, extra: set = EXTRA_KEYS) -> None:
    """Lanza ValueError si el override apunta a una clave que no existe en `cfg_base`."""
    key, _ = _split_override(item)
    conocidas = claves_validas(cfg_base) | set(extra)
    if key not in conocidas:
        raise _error_clave_desconocida(item, key, conocidas)


def _aplicar_override(cfg: dict, item: str) -> None:
    key, value = _split_override(item)
    parsed = _parse(value)
    node = cfg
    *parents, last = key.split(".")
    for k in parents:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override inválido: {item} ('{k}' es un valor, no una sección).")
    if isinstance(node.get(last), dict) and not isinstance(parsed, dict):
        raise ValueError(f"Override '{item}': '{key}' es una sección; sobrescribe una de sus claves "
                         f"(p. ej. {key}.<clave>=...).")
    node[last] = parsed


def load_config(path: str, overrides=None, default_path: str = DEFAULT_PATH, strict: bool = False,
                extra_keys: set = EXTRA_KEYS) -> dict:
    """Carga default.yaml, le superpone `path` y aplica overrides tipo 'train.lr=0.001'.

    strict=True: cada override debe apuntar a una clave existente (o a una de `extra_keys`);
    si no, ValueError con sugerencia. Todas las validaciones se hacen ANTES de aplicar nada,
    contra la config sin overrides, así un override no puede "habilitar" a otro.
    """
    cfg = {}
    if os.path.exists(default_path) and os.path.abspath(path) != os.path.abspath(default_path):
        cfg = _read(default_path)
    cfg = _merge(cfg, _read(path))

    overrides = list(overrides or [])
    if strict:
        for item in overrides:
            validar_override(item, cfg, extra_keys)
    for item in overrides:
        _aplicar_override(cfg, item)
    return cfg