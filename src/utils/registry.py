"""Registro de experimentos: presupuesto de búsqueda, tiempos y parámetros entrenables."""
import csv
import hashlib
import json
import os
from datetime import datetime

import pandas as pd

LOG_PATH = "results/tables/experiment_log.csv"
COLUMNS = ["timestamp", "experiment", "seed", "config_id",
           "n_params", "seconds", "metrics", "config"]

# Claves que NO definen una configuración (no son hiperparámetros del modelo)
_IGNORED_KEYS = {"seed", "run_seeds", "paths"}


def config_id(cfg: dict) -> str:
    """Identificador estable de una configuración: misma config -> mismo id.

    Ignora semillas y rutas, así las 30 corridas de una misma configuración
    cuentan como UNA configuración entrenada.
    """
    clean = {k: v for k, v in cfg.items() if k not in _IGNORED_KEYS}
    raw = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]


def log_run(experiment: str, cfg: dict, seed: int, n_params: int,
            seconds: float, metrics: dict, log_path: str = LOG_PATH) -> None:
    """Agrega una fila al CSV de experimentos (lo crea con encabezado si no existe)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            experiment,
            seed,
            config_id(cfg),
            n_params,
            round(seconds, 3),
            json.dumps(metrics, default=float),
            json.dumps(cfg, default=str),
        ])

def summarize_budget(experiment: str = None, log_path: str = LOG_PATH) -> pd.DataFrame:
    """Resume por experimento: configuraciones distintas, corridas, tiempo total y parámetros."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Aún no hay registro en {log_path}.")
    df = pd.read_csv(log_path)
    if experiment is not None:
        df = df[df["experiment"] == experiment]
    return (df.groupby("experiment")
              .agg(n_configs=("config_id", "nunique"),
                   n_runs=("seed", "count"),
                   total_seconds=("seconds", "sum"),
                   n_params=("n_params", "max"))
              .round(2))