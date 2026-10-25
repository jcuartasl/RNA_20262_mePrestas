"""Argumentos y arranque comunes a todos los experimentos."""
import argparse
from pathlib import Path

from .config import ROOT, load_config
from .seed import set_seed


def build_parser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config,
                   help="YAML del experimento (se mezcla sobre configs/default.yaml)")
    p.add_argument("--override", nargs="*", default=[], metavar="CLAVE=VALOR",
                   help="Cambia valores sin editar el YAML, p. ej. train.lr=0.001 model.hidden=[64,32]")
    return p

def init_run(args) -> dict:
    """Carga la config (con overrides) y fija la semilla global."""
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path          # funciona sin importar desde dónde se ejecute
    cfg = load_config(str(path), args.override)
    set_seed(cfg["seed"])
    return cfg