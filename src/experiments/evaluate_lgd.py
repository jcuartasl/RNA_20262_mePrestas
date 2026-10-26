"""Evalúa las predicciones de LGD guardadas por semilla. Responsable: P4.

Uso:
    python -m src.experiments.evaluate_lgd --config configs/eval_lgd.yaml
    python -m src.experiments.evaluate_lgd --config configs/eval_lgd.yaml --override eval_lgd.figura_seed=5
"""
import time

import numpy as np
import pandas as pd

from src.data import load_splits
from src.evaluation.distributions import (calibracion_lineal, comparar_distribuciones, figura_calibracion_deciles,
                                          figura_distribuciones, tabla_cuantiles)
from src.evaluation.metrics import metricas_lgd
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT


def evaluar_semilla(y, p):
    """Todas las métricas de una semilla: acierto, distribución y calibración."""
    return {**metricas_lgd(y, p), **comparar_distribuciones(y, p), **calibracion_lineal(y, p)}


def resumir(por_semilla):
    """Media, desviación, mínimo y máximo de cada métrica entre semillas, por modelo."""
    metricas = [c for c in por_semilla.columns if c not in ("modelo", "semilla")]
    g = por_semilla.groupby("modelo")[metricas]
    resumen = pd.concat({"media": g.mean(), "sd": g.std(ddof=1), "min": g.min(), "max": g.max()}, axis=1)
    resumen.columns = [f"{metrica}_{estadistico}" for estadistico, metrica in resumen.columns]
    return resumen.join(por_semilla.groupby("modelo").size().rename("n_semillas")).reset_index()


def run(cfg):
    ev = cfg["eval_lgd"]
    datos = load_splits("lgd", ev["scheme"])
    y = np.asarray(datos[f"y_{ev['split']}"], dtype=float)
    filas, preds_figura = [], {}
    for nombre, archivo in ev["predicciones"].items():
        pred = pd.read_csv(ROOT / archivo)
        if len(pred) != len(y):
            raise ValueError(f"{archivo}: {len(pred)} filas, pero y_{ev['split']} tiene {len(y)}")
        for col in pred.columns:
            filas.append({"modelo": nombre, "semilla": int(col.removeprefix("seed")),
                          **evaluar_semilla(y, pred[col].to_numpy())})
        preds_figura[nombre] = pred[f"seed{ev['figura_seed']}"].to_numpy()

    sufijo = f"{ev['split']}_{ev['scheme']}"
    s = ev["figura_seed"]
    tablas, figuras = ROOT / "results" / "tables", ROOT / "results" / "figures"
    por_semilla = pd.DataFrame(filas)
    resumen = resumir(por_semilla)
    por_semilla.to_csv(tablas / f"p4_lgd_eval_{sufijo}_por_semilla.csv", index=False)
    resumen.to_csv(tablas / f"p4_lgd_eval_{sufijo}_resumen.csv", index=False)
    tabla_cuantiles(y, preds_figura).to_csv(tablas / f"p4_lgd_cuantiles_{sufijo}_seed{s}.csv")
    figura_distribuciones(y, preds_figura, figuras / f"p4_lgd_distribucion_{sufijo}_seed{s}.png",
                          titulo=f"LGD observada vs predicha ({ev['split']}, {ev['scheme']}, semilla {s})")
    figura_calibracion_deciles(y, preds_figura, figuras / f"p4_lgd_calibracion_{sufijo}_seed{s}.png",
                               n_bins=ev["n_bins"],
                               titulo=f"Calibración por deciles ({ev['split']}, {ev['scheme']}, semilla {s})")
    return resumen


def main():
    args = build_parser("Evalúa las predicciones de LGD", "configs/eval_lgd.yaml").parse_args()
    cfg = init_run(args)
    inicio = time.perf_counter()
    resumen = run(cfg)
    columnas = ["modelo", "n_semillas", "rmse_media", "rmse_sd", "r2_media", "cociente_sd_media",
                "pendiente_media", "pendiente_sd", "ks_2muestras_media", "obs_sobre_max_pred_media"]
    print(resumen[columnas].round(4).to_string(index=False))
    print(f"[evaluate_lgd] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
