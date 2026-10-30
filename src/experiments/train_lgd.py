"""Entrena la red de LGD (regresión)."""
import time
from pathlib import Path

import numpy as np

from src.data import load_splits
from src.models.lgd_mlp import contar_parametros, entrenar_lgd_mlp, guardar_lgd_mlp, metricas_lgd, predecir_lgd
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT, _merge, load_config
from src.utils.registry import log_run

CONFIG_FINAL = Path(__file__).resolve().parents[2] / "configs" / "lgd_mlp.yaml"


def train(cfg: dict, seed: int) -> dict:
    """Contrato: entrena con `seed` y devuelve {"metrics": {...}, "n_params": int}.

    - Entrena solo con los incumplidos (target_lgd definido).
    - Guarda el modelo con la semilla en el nombre: models/lgd_mlp_seed{seed}.pt
    - Claves de metrics acordadas: rmse, mae, r2.
    """
    # la demo solo trae epochs y sample_fraction: lo demás sale de la config final, no de los valores por defecto
    cfg = _merge(load_config(str(CONFIG_FINAL)), cfg)
    datos = load_splits("lgd", cfg.get("data", {}).get("scheme", "random"))
    X, y = datos["X_train"], datos["y_train"]
    # la demo entrena con una muestra de train para tardar menos de 2 minutos; validación queda completa
    fraccion = cfg.get("sample_fraction", 1.0)
    esquema = cfg.get("data", {}).get("scheme", "random")

    if fraccion < 1:
        elegidos = np.random.default_rng(seed).choice(len(y), int(len(y) * fraccion), replace=False)
        X, y = X[elegidos], y[elegidos]
    # la aleatoria conserva el nombre del contrato; las demás particiones llevan su nombre para no pisarla
    infijo = "" if esquema == "random" else f"_{esquema}"
    nombre = f"lgd_mlp_demo{infijo}_seed{seed}.pt" if fraccion < 1 else f"lgd_mlp{infijo}_seed{seed}.pt"
    modelo, log_phi, _ = entrenar_lgd_mlp(X, y, datos["X_val"], datos["y_val"], cfg, seed)
    guardar_lgd_mlp(modelo, log_phi, cfg, ROOT / "models" / nombre)
    metricas = metricas_lgd(datos["y_val"], predecir_lgd(modelo, datos["X_val"]))
    return {"metrics": metricas, "n_params": contar_parametros(modelo, log_phi)}


def main():
    args = build_parser("Entrena la red de LGD", "configs/lgd_mlp.yaml").parse_args()
    cfg = init_run(args)
    print("[train_lgd] config:", cfg)
    inicio = time.perf_counter()
    salida = train(cfg, cfg["seed"])
    segundos = time.perf_counter() - inicio
    log_run("lgd_mlp", cfg, cfg["seed"], salida["n_params"], segundos, salida["metrics"])
    print(f"[train_lgd] seed={cfg['seed']} {salida['metrics']} n_params={salida['n_params']} "
          f"({segundos:.1f} s)")


if __name__ == "__main__":
    main()
