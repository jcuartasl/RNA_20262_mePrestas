"""
src/expected_loss/el.py
Módulo de cálculo y agregación de Pérdida Esperada:
    E[L] = PD * LGD * EAD
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Dict, Any, Union, Optional
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def expected_loss(
    pd_probs: Union[np.ndarray, pd.Series],
    lgd_values: Union[np.ndarray, pd.Series],
    ead_values: Union[np.ndarray, pd.Series]
) -> Dict[str, Any]:
    """
    Calcula la Pérdida Esperada individual y agregada para una cartera de créditos.

    E[L]_i = PD_i * LGD_i * EAD_i

    Parámetros:
    -----------
    pd_probs : array-like
        Probabilidades de incumplimiento en el intervalo [0, 1].
    lgd_values : array-like
        Pérdida dado el incumplimiento en el intervalo [0, 1].
    ead_values : array-like
        Exposición al incumplimiento (monto financiado en $).

    Retorna:
    --------
    dict con métricas agregadas y vector individual:
        - 'el_individual' : np.ndarray de float32
        - 'el_total'      : float, suma total de pérdidas esperadas ($)
        - 'el_mean'       : float, pérdida esperada promedio por crédito ($)
        - 'el_median'     : float, pérdida esperada mediana ($)
        - 'ead_total'     : float, exposición total agregada de la cartera ($)
        - 'el_rate'       : float, tasa de pérdida esperada (el_total / ead_total)
    """
    pd_arr = np.asarray(pd_probs, dtype=np.float32)
    lgd_arr = np.asarray(lgd_values, dtype=np.float32)
    ead_arr = np.asarray(ead_values, dtype=np.float32)

    # Validar dimensiones
    if not (len(pd_arr) == len(lgd_arr) == len(ead_arr)):
        raise ValueError(
            f"Inconsistencia en dimensiones: pd={len(pd_arr)}, lgd={len(lgd_arr)}, ead={len(ead_arr)}"
        )

    # Cálculo elemento a elemento
    el_individual = pd_arr * lgd_arr * ead_arr

    el_total = float(np.sum(el_individual))
    ead_total = float(np.sum(ead_arr))
    el_mean = float(np.mean(el_individual))
    el_median = float(np.median(el_individual))
    el_rate = float(el_total / ead_total) if ead_total > 0 else 0.0

    return {
        "el_individual": el_individual,
        "el_total": el_total,
        "el_mean": el_mean,
        "el_median": el_median,
        "ead_total": ead_total,
        "el_rate": el_rate,
    }


def expected_loss_report(
    pd_probs: Union[np.ndarray, pd.Series],
    lgd_values: Union[np.ndarray, pd.Series],
    ead_values: Union[np.ndarray, pd.Series],
    figures_dir: Optional[str] = "results/figures"
) -> Dict[str, Any]:
    """
    Ejecuta el cálculo de E[L], imprime reporte formateado y genera
    histograma de distribución y boxplot por deciles de PD.
    """
    results = expected_loss(pd_probs, lgd_values, ead_values)
    el_ind = results["el_individual"]
    pd_arr = np.asarray(pd_probs, dtype=np.float32)

    print("\n" + "="*60)
    print("REPORTE DE PÉRDIDA ESPERADA (EXPECTED LOSS - E[L]):")
    print("="*60)
    print(f"Créditos evaluados:                 {len(el_ind):,}")
    print(f"Exposición total (EAD Total):       ${results['ead_total']:,.2f}")
    print(f"Pérdida Esperada Total (E[L]):      ${results['el_total']:,.2f}")
    print(f"Tasa de Pérdida Esperada (E[L]/EAD): {results['el_rate']*100:.2f}%")
    print(f"Pérdida Esperada Promedio/crédito:  ${results['el_mean']:,.2f}")
    print(f"Pérdida Esperada Mediana/crédito:   ${results['el_median']:,.2f}")
    print("="*60)

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)

        # 1. Histograma de distribución de E[L]
        plt.figure(figsize=(9, 5))
        plt.hist(el_ind, bins=50, color="#1f77b4", edgecolor="black", alpha=0.8)
        plt.title("Distribución de Pérdida Esperada Individual E[L]", fontsize=13, fontweight="bold")
        plt.xlabel("Pérdida Esperada ($ USD)", fontsize=11)
        plt.ylabel("Frecuencia", fontsize=11)
        plt.axvline(results["el_mean"], color="red", linestyle="--", linewidth=1.5, label=f"Media: ${results['el_mean']:,.1f}")
        plt.axvline(results["el_median"], color="black", linestyle=":", linewidth=1.5, label=f"Mediana: ${results['el_median']:,.1f}")
        plt.legend(frameon=True)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        fig_dist = os.path.join(figures_dir, "distribucion_el.png")
        plt.savefig(fig_dist, dpi=300)
        plt.close()
        print(f"Gráfica guardada: {fig_dist}")

        # 2. Boxplot / barras de E[L] por rangos de PD
        bins = np.linspace(0.0, 1.0, 11)
        labels = [f"{int(bins[i]*100)}-{int(bins[i+1]*100)}%" for i in range(len(bins)-1)]
        pd_binned = pd.cut(pd_arr, bins=bins, labels=labels, include_lowest=True)
        
        df_temp = pd.DataFrame({"rango_pd": pd_binned, "el": el_ind})
        el_by_bin = df_temp.groupby("rango_pd", observed=False)["el"].mean()

        plt.figure(figsize=(10, 5))
        el_by_bin.plot(kind="bar", color="#ff7f0e", edgecolor="black", alpha=0.85)
        plt.title("Pérdida Esperada Media por Rango de Probabilidad de Incumplimiento (PD)", fontsize=13, fontweight="bold")
        plt.xlabel("Rango de PD (%)", fontsize=11)
        plt.ylabel("Pérdida Esperada Promedio ($ USD)", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig_box = os.path.join(figures_dir, "el_por_rango_pd.png")
        plt.savefig(fig_box, dpi=300)
        plt.close()
        print(f"Gráfica guardada: {fig_box}")

    return results


if __name__ == "__main__":
    pds = np.array([0.05, 0.20, 0.80], dtype=np.float32)
    lgds = np.array([0.50, 0.60, 0.70], dtype=np.float32)
    eads = np.array([10000.0, 5000.0, 20000.0], dtype=np.float32)
    res = expected_loss(pds, lgds, eads)
    print("Prueba unitaria EL:", res)
