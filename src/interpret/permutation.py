"""Importancia por permutación agrupada por variable original. Responsable: P4.

Breiman (2001); Fisher, Rudin y Dominici (2019). Importancia = métrica_base - métrica con la
variable permutada. Las columnas de una misma variable (p. ej. las de one-hot) se permutan con
la MISMA reordenación de filas, así cada fila sigue teniendo una sola categoría activa.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import auc


def grupos_por_variable(columnas, variables_originales, grupos_extra=None):
    """Asigna cada columna procesada a su variable original.

    Coincidencia exacta primero (numéricas); si no, el prefijo 'variable_' más largo (categóricas).
    grupos_extra: dict nombre -> lista de variables originales que solo se permutan juntas.
    """
    grupos = {v: [] for v in variables_originales}
    for i, col in enumerate(columnas):
        if col in grupos:
            grupos[col].append(i)
            continue
        candidatas = [v for v in variables_originales if col.startswith(v + "_")]
        if not candidatas:
            raise ValueError(f"no se encontró la variable original de la columna {col!r}")
        grupos[max(candidatas, key=len)].append(i)
    grupos = {v: idx for v, idx in grupos.items() if idx}
    # Los miembros de un grupo extra se REEMPLAZAN por el grupo: permutar por separado variables casi
    # idénticas crea filas imposibles y obliga al modelo a extrapolar (Hooker et al., 2021).
    for nombre, miembros in (grupos_extra or {}).items():
        grupos[nombre] = sorted(i for m in miembros for i in grupos.pop(m))
    return grupos


def importancia_permutacion(predecir, X, y, grupos, metrica=auc, n_repeticiones=10, seed=90):
    """predecir: X -> puntaje de riesgo (PD o logit; mayor = más riesgo)."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    pred_base = np.asarray(predecir(X), dtype=float)
    base = metrica(y, pred_base)
    filas = []
    for nombre, idx in grupos.items():
        caidas = []
        for _ in range(n_repeticiones):
            Xp = X.copy()
            orden = rng.permutation(X.shape[0])
            Xp[:, idx] = X[orden][:, idx]
            caidas.append(base - metrica(y, np.asarray(predecir(Xp), dtype=float)))
        direccion = np.nan
        if len(idx) == 1 and np.std(X[:, idx[0]]) > 0:
            direccion = float(np.corrcoef(X[:, idx[0]], pred_base)[0, 1])
        filas.append({"variable": nombre, "n_columnas": len(idx), "importancia_media": float(np.mean(caidas)),
                      "importancia_sd": float(np.std(caidas, ddof=1)), "direccion_corr": direccion})
    tabla = pd.DataFrame(filas).sort_values("importancia_media", ascending=False).reset_index(drop=True)
    tabla["metrica_base"] = base
    return tabla


def figura_importancia(tabla, ruta, titulo, top=20, etiqueta="caída de AUC al permutar (media ± sd)"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tabla.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(t) + 1.5))
    ax.barh(t["variable"], t["importancia_media"], xerr=t["importancia_sd"], alpha=0.8)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set(xlabel=etiqueta, title=titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
