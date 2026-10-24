"""
notebooks/analisis_temporal.py
Análisis de estabilidad temporal, tasa de default anual, sesgo de madurez
y dinámica de LGD / EAD (Fase 4 para P1).
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    figures_dir = os.path.join(project_root, "results", "figures")
    processed_dir = os.path.join(project_root, "data", "processed")
    raw_csv = os.path.join(project_root, "data", "raw", "loan.csv")

    os.makedirs(figures_dir, exist_ok=True)

    print("=================================================================")
    print("FASE 4: ANÁLISIS DE ESTABILIDAD TEMPORAL Y SESGO DE MADUREZ")
    print("=================================================================")

    # 1. Tasa de default por año de originación (2007-2015) en PD resueltos
    print("\n[1/4] Analizando tasa de default anual en créditos resueltos...")
    df_pd = pd.read_parquet(os.path.join(processed_dir, "loans_clean_pd.parquet"))
    
    anual_pd = df_pd.groupby("issue_year").agg(
        total_creditos=("target_pd", "count"),
        defaults=("target_pd", "sum"),
        tasa_default=("target_pd", "mean")
    ).reset_index()

    print(anual_pd.to_string(index=False))

    # Graficar tasa de default por año
    plt.figure(figsize=(9, 5))
    bars = plt.bar(anual_pd["issue_year"].astype(str), anual_pd["tasa_default"] * 100, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.title("Tasa de Incumplimiento Observada por Año de Originación (PD)", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación (issue_year)", fontsize=11)
    plt.ylabel("Tasa de Default (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(0, max(anual_pd["tasa_default"] * 100) + 5)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "tasa_default_por_anio.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"      Figura guardada: {fig1_path}")

    # 2. Análisis del sesgo de madurez (Créditos madurados vs vigentes en dataset crudo)
    print("\n[2/4] Analizando sesgo de madurez desde dataset crudo (loan.csv)...")
    # Leer columnas necesarias del crudo
    df_raw = pd.read_csv(raw_csv, usecols=["issue_d", "loan_status"], low_memory=False)
    df_raw["issue_year"] = pd.to_datetime(df_raw["issue_d"], format="%b-%Y", errors="coerce").dt.year

    madurez_df = df_raw.groupby("issue_year").apply(
        lambda g: pd.Series({
            "total": len(g),
            "vigentes_current": (g["loan_status"] == "Current").sum(),
            "pct_vigente": (g["loan_status"] == "Current").mean() * 100,
            "pct_resuelto": (g["loan_status"].isin([
                "Fully Paid", "Charged Off", "Default", 
                "Does not meet the credit policy. Status:Fully Paid",
                "Does not meet the credit policy. Status:Charged Off"
            ])).mean() * 100
        }),
        include_groups=False
    ).reset_index()

    print(madurez_df.to_string(index=False))

    # Graficar % madurados vs % vigentes
    plt.figure(figsize=(10, 5))
    x = np.arange(len(madurez_df))
    width = 0.4

    plt.bar(x - width/2, madurez_df["pct_resuelto"], width, label="Créditos Resueltos (%)", color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.bar(x + width/2, madurez_df["pct_vigente"], width, label="Créditos Vigentes 'Current' (%)", color="#ff7f0e", edgecolor="black", alpha=0.85)

    plt.xticks(x, madurez_df["issue_year"].astype(int))
    plt.title("Sesgo de Madurez por Año: Proporción de Créditos Resueltos vs. Vigentes", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación", fontsize=11)
    plt.ylabel("Porcentaje del Total Otorgado (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "sesgo_madurez.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"      Figura guardada: {fig2_path}")

    # 3. Distribución temporal de LGD en incumplidos
    print("\n[3/4] Analizando LGD promedio por año de originación...")
    df_lgd = pd.read_parquet(os.path.join(processed_dir, "loans_clean_lgd.parquet"))
    
    anual_lgd = df_lgd.groupby("issue_year").agg(
        defaults_lgd=("lgd", "count"),
        lgd_mean=("lgd", "mean"),
        lgd_median=("lgd", "median")
    ).reset_index()

    print(anual_lgd.to_string(index=False))

    plt.figure(figsize=(9, 5))
    bars = plt.bar(anual_lgd["issue_year"].astype(str), anual_lgd["lgd_mean"] * 100, color="#d62728", edgecolor="black", alpha=0.85)
    plt.title("LGD Media por Año de Originación en Créditos Incumplidos", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación (issue_year)", fontsize=11)
    plt.ylabel("LGD Media (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(0, 105)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "lgd_media_por_anio.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"      Figura guardada: {fig3_path}")

    # 4. Distribución de montos (EAD) por año
    print("\n[4/4] Analizando EAD medio financiado por año...")
    anual_ead = df_pd.groupby("issue_year").agg(
        ead_mean=("funded_amnt", "mean"),
        ead_median=("funded_amnt", "median")
    ).reset_index()

    print(anual_ead.to_string(index=False))

    plt.figure(figsize=(9, 5))
    plt.plot(anual_ead["issue_year"].astype(str), anual_ead["ead_mean"], marker="o", linewidth=2.5, color="#9467bd", label="EAD Medio ($)")
    plt.plot(anual_ead["issue_year"].astype(str), anual_ead["ead_median"], marker="s", linestyle="--", linewidth=2, color="#8c564b", label="EAD Mediana ($)")
    plt.title("Evolución del Monto Financiado (EAD) por Año de Originación", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación", fontsize=11)
    plt.ylabel("Monto ($ USD)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "ead_medio_por_anio.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"      Figura guardada: {fig4_path}")

    print("\nAnálisis temporal completado con éxito.")


if __name__ == "__main__":
    main()
