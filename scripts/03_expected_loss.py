"""
scripts/03_expected_loss.py
Cálculo de la Pérdida Esperada de referencia (baseline / naive) sobre el conjunto de TEST temporal.
PD naive: tasa de default observada en test.
LGD naive: media observada en el universo de train de LGD.
EAD: monto financiado real (funded_amnt).

Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

import os
import sys
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.expected_loss.el import expected_loss_report


def main():
    processed_dir = os.path.join(project_root, "data", "processed")
    figures_dir = os.path.join(project_root, "results", "figures")

    print("=================================================================")
    print("FASE 5: PÉRDIDA ESPERADA DE REFERENCIA (BASELINE NAIVE)")
    print("=================================================================")

    # 1. Cargar datos limpios para extraer EAD real y partición de Test
    df_pd = pd.read_parquet(os.path.join(processed_dir, "loans_clean_pd.parquet"))
    df_lgd = pd.read_parquet(os.path.join(processed_dir, "loans_clean_lgd.parquet"))

    # Partición temporal de Test (2015)
    test_pd = df_pd[df_pd["issue_year"] == 2015].copy()
    train_lgd = df_lgd[df_lgd["issue_year"] <= 2013].copy()

    n_test = len(test_pd)
    print(f"\n[1/3] Conjunto de Test Temporal (2015): {n_test:,} créditos evaluados.")

    # 2. Configuración de estimadores de referencia (naive)
    # Tasa observada de default en test
    pd_naive_rate = float(test_pd["target_pd"].mean())
    pd_naive = np.full(n_test, fill_value=pd_naive_rate, dtype=np.float32)

    # LGD media observada en train
    lgd_naive_rate = float(train_lgd["lgd"].mean())
    lgd_naive = np.full(n_test, fill_value=lgd_naive_rate, dtype=np.float32)

    # EAD real al originar
    ead_real = test_pd["funded_amnt"].to_numpy(dtype=np.float32)

    print(f"[2/3] Estimadores naive aplicados:")
    print(f"      - PD naive (tasa media 2015):     {pd_naive_rate*100:.2f}%")
    print(f"      - LGD naive (media train <=2013): {lgd_naive_rate*100:.2f}%")
    print(f"      - EAD total en cartera test:      ${np.sum(ead_real):,.2f}")

    # 3. Ejecutar cálculo y reporte
    print("\n[3/3] Generando reporte y figuras de Pérdida Esperada...")
    el_results = expected_loss_report(
        pd_probs=pd_naive,
        lgd_values=lgd_naive,
        ead_values=ead_real,
        figures_dir=figures_dir
    )

    print("\nNota metodológica:")
    print("Estos valores de Pérdida Esperada naive sirven como cota de referencia para el reporte.")
    print("Posteriormente, cuando P2 (MLP de PD) y P3 (MLP de LGD) generen las probabilidades y fracciones")
    print("predichas individuales, se sustituirán estos vectores manteniendo la misma función expected_loss().\n")


if __name__ == "__main__":
    main()
