"""Comparación entre la distribución observada y la predicha (LGD). Responsable: P4."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from src.evaluation.metrics import _preparar

CUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def comparar_distribuciones(y, p):
    """Resumen de qué tan parecida es la distribución de p a la de y (no mide acierto crédito a crédito)."""
    y, p = _preparar(y, p)
    return {
        "media_obs": float(y.mean()), "media_pred": float(p.mean()),
        "sd_obs": float(y.std(ddof=1)), "sd_pred": float(p.std(ddof=1)),
        "cociente_sd": float(p.std(ddof=1) / y.std(ddof=1)),
        "ks_2muestras": float(ks_2samp(y, p).statistic),
        "wasserstein": float(wasserstein_distance(y, p)),
        "min_pred": float(p.min()), "max_pred": float(p.max()),
        "obs_bajo_min_pred": float(np.mean(y < p.min())),
        "obs_sobre_max_pred": float(np.mean(y > p.max())),
    }


def tabla_cuantiles(y, predicciones, cuantiles=CUANTILES):
    """Cuantiles de la observada y de cada modelo. predicciones: dict nombre -> arreglo."""
    columnas = {"observada": np.quantile(np.asarray(y, dtype=float), cuantiles)}
    for nombre, p in predicciones.items():
        _preparar(y, p)
        columnas[nombre] = np.quantile(np.asarray(p, dtype=float), cuantiles)
    return pd.DataFrame(columnas, index=pd.Index(cuantiles, name="cuantil"))


def figura_distribuciones(y, predicciones, ruta, titulo="LGD observada vs predicha"):
    """Histograma superpuesto (izquierda) y gráfico QQ predicha vs observada (derecha)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(y, dtype=float)
    fig, (ax_h, ax_q) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 41)
    ax_h.hist(y, bins=bins, density=True, alpha=0.45, color="gray", label="observada")
    q = np.linspace(0.005, 0.995, 199)
    q_obs = np.quantile(y, q)
    for nombre, p in predicciones.items():
        p = np.asarray(p, dtype=float)
        ax_h.hist(p, bins=bins, density=True, histtype="step", linewidth=1.8, label=nombre)
        ax_q.plot(q_obs, np.quantile(p, q), linewidth=1.8, label=nombre)
    ax_q.plot([0, 1], [0, 1], "k--", linewidth=1, label="identidad")
    ax_h.set(xlabel="LGD", ylabel="densidad", title="Histogramas")
    ax_q.set(xlabel="cuantiles observados", ylabel="cuantiles predichos", title="Gráfico QQ",
             xlim=(0, 1), ylim=(0, 1))
    ax_h.legend(); ax_q.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def calibracion_lineal(y, p):
    """Regresión de y sobre p. Calibrado: pendiente ~ 1 e intercepto ~ 0.

    pendiente < 1: predicciones demasiado dispersas; pendiente > 1: demasiado comprimidas.
    """
    y, p = _preparar(y, p)
    pendiente, intercepto = np.polyfit(p, y, 1)
    return {"pendiente": float(pendiente), "intercepto": float(intercepto)}


def calibracion_deciles(y, p, n_bins=10):
    """Por bin de cuantiles de p: n, predicción media y valor observado medio (variable continua)."""
    y, p = _preparar(y, p)
    bordes = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    bin_id = np.digitize(p, bordes[1:-1], right=True)
    return (
        pd.DataFrame({"bin": bin_id, "p": p, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), pred_media=("p", "mean"), obs_media=("y", "mean"))
        .reset_index()
    )


def figura_calibracion_deciles(y, predicciones, ruta, n_bins=10, titulo="Calibración por deciles"):
    """Media observada vs predicha por bin de cuantiles de la predicción, para cada modelo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5))
    valores = []
    for nombre, p in predicciones.items():
        t = calibracion_deciles(y, p, n_bins)
        ax.plot(t["pred_media"], t["obs_media"], marker="o", linewidth=1.5, label=nombre)
        valores += list(t["pred_media"]) + list(t["obs_media"])
    lo, hi = min(valores) - 0.02, max(valores) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="identidad")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="predicción media del bin",
           ylabel="valor observado medio del bin", title=titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta
