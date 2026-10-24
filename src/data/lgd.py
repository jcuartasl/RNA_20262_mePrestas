"""
src/data/lgd.py
Módulo para construcción del objetivo continuo de Pérdida Dado el Incumplimiento (LGD)
y Exposición al Incumplimiento (EAD).
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Tuple, Dict, Any
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_lgd(
    df: pd.DataFrame,
    plot_histogram: bool = True,
    figure_path: str = "results/figures/distribucion_lgd.png"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Construye el conjunto de datos para la Parte 2 (LGD) filtrando el universo de incumplidos (target_pd == 1)
    y calculando LGD y EAD bajo la Opción A (EAD al originar):

        EAD = funded_amnt
        perdida = funded_amnt - total_rec_prncp - (recoveries - collection_recovery_fee)
        LGD = clip(perdida / funded_amnt, 0, 1)

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame limpio que contiene 'target_pd' y las columnas de recuperación y montos.
    plot_histogram : bool, default=True
        Si es True, genera y guarda la figura de distribución de LGD.
    figure_path : str
        Ruta donde guardar la gráfica generada.

    Retorna:
    --------
    lgd_df : pd.DataFrame
        Subconjunto filtrado para target_pd == 1 con las columnas 'lgd' y 'ead'.
    stats : dict
        Estadísticas descriptivas de la variable LGD.
    """
    if "target_pd" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'target_pd' codificada (0/1).")

    # 1. Filtrar universo de créditos incumplidos (target_pd == 1)
    defaults = df[df["target_pd"] == 1].copy()

    # Validar presencia de columnas necesarias para la fórmula Opción A
    required_cols = ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]
    for col in required_cols:
        if col not in defaults.columns:
            raise KeyError(f"Columna requerida '{col}' no encontrada en el DataFrame.")

    # 2. Imputar nulos eventuales con 0.0 para las variables de recuperación
    funded = defaults["funded_amnt"].fillna(0.0)
    principal_rec = defaults["total_rec_prncp"].fillna(0.0)
    recoveries = defaults["recoveries"].fillna(0.0)
    fee = defaults["collection_recovery_fee"].fillna(0.0)

    # 3. Aplicar fórmula de pérdida neta
    # Recuperación neta = recoveries - collection_recovery_fee
    net_recoveries = recoveries - fee
    perdida = funded - principal_rec - net_recoveries

    # LGD como fracción de la exposición original al originar
    raw_lgd = perdida / np.where(funded > 0, funded, np.nan)

    # Recorte estricto al intervalo [0, 1]
    defaults["lgd"] = np.clip(raw_lgd, 0.0, 1.0)
    defaults["ead"] = funded

    # 4. Cálculo de estadísticas descriptivas
    lgd_series = defaults["lgd"]
    n_total = len(lgd_series)
    pct_zeros = float((lgd_series == 0.0).mean() * 100)
    pct_ones = float((lgd_series == 1.0).mean() * 100)

    stats = {
        "n_defaults": n_total,
        "mean": float(lgd_series.mean()),
        "std": float(lgd_series.std()),
        "median": float(lgd_series.median()),
        "min": float(lgd_series.min()),
        "max": float(lgd_series.max()),
        "pct_exact_zero": pct_zeros,
        "pct_exact_one": pct_ones,
        "raw_below_zero_count": int((raw_lgd < 0.0).sum()),
        "raw_above_one_count": int((raw_lgd > 1.0).sum()),
    }

    # 5. Generación de histograma de distribución
    if plot_histogram:
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.figure(figsize=(9, 5))
        plt.hist(lgd_series, bins=50, color="#d62728", edgecolor="black", alpha=0.75, density=True)
        plt.title("Distribución Empírica de LGD en Créditos Incumplidos (Opción A)", fontsize=13, fontweight="bold")
        plt.xlabel("Loss Given Default (LGD)", fontsize=11)
        plt.ylabel("Densidad", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)

        # Líneas de referencia
        plt.axvline(lgd_series.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Media: {lgd_series.mean():.3f}")
        plt.axvline(lgd_series.median(), color="blue", linestyle=":", linewidth=1.5, label=f"Mediana: {lgd_series.median():.3f}")
        plt.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(figure_path, dpi=300)
        plt.close()

    return defaults, stats


if __name__ == "__main__":
    dummy_data = pd.DataFrame({
        "target_pd": [1, 1, 0, 1],
        "funded_amnt": [10000.0, 5000.0, 8000.0, 2000.0],
        "total_rec_prncp": [2000.0, 0.0, 8000.0, 200.0],
        "recoveries": [500.0, 0.0, 0.0, 100.0],
        "collection_recovery_fee": [50.0, 0.0, 0.0, 10.0]
    })
    lgd_res, st = compute_lgd(dummy_data, plot_histogram=False)
    print("Muestra LGD calculada:")
    print(lgd_res[["funded_amnt", "total_rec_prncp", "lgd", "ead"]])
    print("Estadísticas:", st)
