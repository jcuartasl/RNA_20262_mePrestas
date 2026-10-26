"""Evaluación de PD por semilla: logística (P3) frente a red (P2). Responsable: P4.

Uso:
    python -m src.experiments.evaluate_pd --config configs/eval_pd.yaml
    python -m src.experiments.evaluate_pd --config configs/eval_pd.yaml --override eval_pd.split=val
Cuando P2 suba sus 30 modelos o sus predicciones, se agregan en configs/eval_pd.yaml.
"""
import re
import time

import numpy as np
import pandas as pd
from scipy.special import expit

from src.data import load_splits
from src.evaluation.metrics import curva_calibracion, metricas_pd
from src.experiments.build_scorecard import cargar_modelo_pd
from src.experiments.evaluate_lgd import resumir
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT

TABLAS, FIGURAS = ROOT / "results" / "tables", ROOT / "results" / "figures"


def _columna(ruta, i):
    m = re.search(r"seed(\d+)", str(ruta))
    return f"seed{m.group(1)}" if m else f"seed{i}"


def predecir_modelos(nombre, tipo, rutas, X, sufijo):
    """Predice con cada archivo de modelo y guarda results/tables/pd_pred_<sufijo>_<nombre>.csv.gz."""
    columnas = {}
    for i, ruta in enumerate(rutas):
        logit, _, _ = cargar_modelo_pd({"modelo": tipo, "ruta_modelo": ruta})
        columnas[_columna(ruta, i)] = expit(np.asarray(logit(X), dtype=float))
    pred = pd.DataFrame(columnas)
    pred.to_csv(TABLAS / f"pd_pred_{sufijo}_{nombre}.csv.gz", index=False)
    return pred


def figura_calibracion(y, preds, ruta, n_bins, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    hi = 0.0
    for nombre, pred in preds.items():
        t = curva_calibracion(y, pred.iloc[:, 0].to_numpy(), n_bins=n_bins)
        ax.plot(t["pd_media"], t["tasa_observada"], marker="o", label=f"{nombre} ({pred.columns[0]})")
        hi = max(hi, t["pd_media"].max(), t["tasa_observada"].max())
    ax.plot([0, hi + 0.02], [0, hi + 0.02], "k--", linewidth=1, label="identidad")
    ax.set(xlabel="PD media predicha del bin", ylabel="tasa de incumplimiento observada", title=titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def run(cfg):
    ev = cfg["eval_pd"]
    d = load_splits("pd", ev["scheme"])
    X, y = d[f"X_{ev['split']}"], np.asarray(d[f"y_{ev['split']}"])
    sufijo = f"{ev['split']}_{ev['scheme']}"
    preds = {nombre: predecir_modelos(nombre, m["tipo"], m["rutas"], X, sufijo)
             for nombre, m in (ev.get("modelos") or {}).items()}
    for nombre, archivo in (ev.get("predicciones") or {}).items():
        pred = pd.read_csv(ROOT / archivo)
        if len(pred) != len(y):
            raise ValueError(f"{archivo}: {len(pred)} filas, pero y_{ev['split']} tiene {len(y)}")
        preds[nombre] = pred

    filas = []
    for nombre, pred in preds.items():
        for col in pred.columns:
            p = pred[col].to_numpy()
            filas.append({"modelo": nombre, "semilla": int(col.removeprefix("seed")), **metricas_pd(y, p),
                          "pd_media": float(p.mean()), "tasa_real": float(y.mean())})
    por_semilla = pd.DataFrame(filas)
    resumen = resumir(por_semilla)
    por_semilla.to_csv(TABLAS / f"p4_pd_eval_{sufijo}_por_semilla.csv", index=False)
    resumen.to_csv(TABLAS / f"p4_pd_eval_{sufijo}_resumen.csv", index=False)
    figura_calibracion(y, preds, FIGURAS / f"p4_pd_calibracion_{sufijo}.png", int(ev["n_bins"]),
                       f"Calibración de PD ({ev['split']}, {ev['scheme']})")
    return resumen


def main():
    args = build_parser("Evaluación de PD", "configs/eval_pd.yaml").parse_args()
    cfg = init_run(args)
    inicio = time.perf_counter()
    resumen = run(cfg)
    cols = ["modelo", "n_semillas", "auc_media", "auc_sd", "gini_media", "ks_media", "brier_media", "bce_media",
            "pd_media_media", "tasa_real_media"]
    print(resumen[cols].round(4).to_string(index=False))
    print(f"[evaluate_pd] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
