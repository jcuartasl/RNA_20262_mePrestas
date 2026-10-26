"""Métricas de evaluación para PD (clasificación) y LGD (regresión). Responsable: P4.

Contrato: cada función recibe y (observado) y p (predicho) como arreglos 1D.
metricas_pd y metricas_lgd devuelven diccionarios con las claves acordadas:
    PD:  auc, gini, ks, brier, bce
    LGD: rmse, mae, r2
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

EPS = np.finfo(np.float64).eps


# ---------------------------------------------------------------- validación

def _preparar(y, p):
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y y p deben tener el mismo tamaño: {y.shape} vs {p.shape}")
    if y.size == 0:
        raise ValueError("y y p están vacíos")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("y y p no pueden contener NaN ni infinitos")
    return y, p


def _preparar_binaria(y, p):
    y, p = _preparar(y, p)
    if not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("y debe ser binaria (0 = buen pagador, 1 = incumplimiento)")
    if y.min() == y.max():
        raise ValueError("y debe contener ambas clases")
    return y, p


def _validar_probabilidades(p):
    if p.min() < 0.0 or p.max() > 1.0:
        raise ValueError("p debe estar en [0, 1]")


# ---------------------------------------------------------------- PD

def auc(y, p):
    """AUC vía Mann-Whitney: P(p_malo > p_bueno), con empates contados como 1/2."""
    y, p = _preparar_binaria(y, p)
    rangos = rankdata(p)  # rango promedio en empates
    n1 = y.sum()
    n0 = y.size - n1
    return float((rangos[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gini(y, p):
    """Índice de Gini (accuracy ratio): 2 * AUC - 1."""
    return 2.0 * auc(y, p) - 1.0


def ks(y, p):
    """Estadístico KS: máxima distancia entre las FDA empíricas de p en malos y en buenos."""
    y, p = _preparar_binaria(y, p)
    malos = np.sort(p[y == 1])
    buenos = np.sort(p[y == 0])
    umbrales = np.unique(p)
    f_malos = np.searchsorted(malos, umbrales, side="right") / malos.size
    f_buenos = np.searchsorted(buenos, umbrales, side="right") / buenos.size
    return float(np.max(np.abs(f_malos - f_buenos)))


def brier(y, p):
    """Puntaje de Brier: error cuadrático medio entre la probabilidad y el resultado."""
    y, p = _preparar_binaria(y, p)
    _validar_probabilidades(p)
    return float(np.mean((p - y) ** 2))


def bce(y, p):
    """Entropía cruzada binaria (log-verosimilitud negativa media de Bernoulli)."""
    y, p = _preparar_binaria(y, p)
    _validar_probabilidades(p)
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def curva_calibracion(y, p, n_bins=10, estrategia="cuantil"):
    """Tabla de calibración: por bin, n, PD media predicha y tasa observada de incumplimiento.

    estrategia="cuantil": bins con igual número de créditos (recomendado con clases desbalanceadas).
    estrategia="uniforme": bins de igual ancho en [0, 1].
    """
    y, p = _preparar_binaria(y, p)
    _validar_probabilidades(p)
    if estrategia == "cuantil":
        bordes = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    elif estrategia == "uniforme":
        bordes = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("estrategia debe ser 'cuantil' o 'uniforme'")
    bordes = np.unique(bordes)  # colapsa bordes repetidos cuando hay muchos empates
    bin_id = np.digitize(p, bordes[1:-1], right=True)
    tabla = (
        pd.DataFrame({"bin": bin_id, "p": p, "y": y})
        .groupby("bin")
        .agg(n=("y", "size"), pd_media=("p", "mean"), tasa_observada=("y", "mean"))
        .reset_index()
    )
    return tabla


def metricas_pd(y, p):
    """Métricas de discriminación y calibración para PD."""
    a = auc(y, p)
    return {"auc": a, "gini": 2.0 * a - 1.0, "ks": ks(y, p), "brier": brier(y, p), "bce": bce(y, p)}


# ---------------------------------------------------------------- LGD

def rmse(y, p):
    y, p = _preparar(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    y, p = _preparar(y, p)
    return float(np.mean(np.abs(y - p)))


def r2(y, p):
    y, p = _preparar(y, p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("r2 no está definido si y es constante")
    return float(1.0 - np.sum((y - p) ** 2) / ss_tot)


def metricas_lgd(y, p):
    """Métricas de regresión para LGD."""
    return {"rmse": rmse(y, p), "mae": mae(y, p), "r2": r2(y, p)}
