"""
scripts/01_clean_and_prepare.py
Script de ejecución completa para Fase 1a:
1. Carga data/raw/loan.csv
2. Limpieza y codificación de PD
3. Construcción de objetivo LGD y EAD (antes de remover columnas de recuperación)
4. Auditoría y remoción de fuga de datos (data leakage)
5. Guardado de datasets limpios en Parquet en data/processed/

Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

import os
import sys
import time
import pandas as pd

# Asegurar acceso a módulos de src/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.clean import clean_loans
from src.data.leakage import remove_leakage, generate_leakage_report
from src.data.lgd import compute_lgd


def main():
    start_time = time.time()
    raw_path = os.path.join(project_root, "data", "raw", "loan.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    tables_dir = os.path.join(project_root, "results", "tables")
    figures_dir = os.path.join(project_root, "results", "figures")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("=================================================================")
    print("FASE 1a: LIMPIEZA, OBJETIVOS (PD/LGD) Y AUDITORÍA DE LEAKAGE")
    print("=================================================================")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos original en: {raw_path}")

    # 1. Cargar archivo original
    print(f"\n[1/5] Cargando dataset original desde {raw_path}...")
    t0 = time.time()
    df_raw = pd.read_csv(raw_path, low_memory=False)
    print(f"      Carga completada en {time.time() - t0:.2f} s. Filas: {len(df_raw):,}, Columnas: {df_raw.shape[1]}")

    # 2. Limpieza inicial y codificación de target_pd
    print("\n[2/5] Ejecutando clean_loans(): codificando PD y excluyendo estados indeterminados...")
    df_clean, clean_report = clean_loans(df_raw, null_threshold=0.50)
    print(f"      Registros resueltos para modelado: {clean_report['n_rows_resolved']:,} "
          f"({clean_report['n_rows_resolved']/clean_report['n_rows_initial']*100:.2f}% del original)")
    print(f"      Excluidos (NA / vigentes / sin historia): {clean_report['n_rows_excluded']:,}")
    print(f"      Buenos pagadores (0): {clean_report['n_non_defaults']:,}")
    print(f"      Incumplidos (1): {clean_report['n_defaults']:,}")
    print(f"      Tasa de default global observada: {clean_report['default_rate']*100:.2f}%")
    print(f"      Columnas descartadas por > 50% nulos: {len(clean_report['high_null_cols_dropped'])}")

    # 3. Construcción del objetivo LGD (Parte 2) antes de eliminar variables de recuperación
    print("\n[3/5] Ejecutando compute_lgd() sobre universo de incumplidos...")
    df_lgd, lgd_stats = compute_lgd(
        df_clean,
        plot_histogram=True,
        figure_path=os.path.join(figures_dir, "distribucion_lgd.png")
    )
    print(f"      Universo de créditos en default para LGD: {lgd_stats['n_defaults']:,}")
    print(f"      LGD Media: {lgd_stats['mean']:.4f} | Mediana: {lgd_stats['median']:.4f} | Std: {lgd_stats['std']:.4f}")
    print(f"      LGD = 0.0 (recuperación total): {lgd_stats['pct_exact_zero']:.2f}%")
    print(f"      LGD = 1.0 (pérdida total): {lgd_stats['pct_exact_one']:.2f}%")

    # 4. Generar reporte de fuga de información y eliminar variables prohibidas
    print("\n[4/5] Generando reporte de fuga de información y aplicando filtro estricto...")
    leakage_csv = os.path.join(tables_dir, "leakage_report.csv")
    generate_leakage_report(df_clean, output_path=leakage_csv)

    # Remover fuga de información (strict=True: elimina scoring propio grade/sub_grade/int_rate/installment y variables post-originación)
    df_pd_clean = remove_leakage(df_clean, strict=True)
    # Para LGD, eliminamos también la fuga post-originación, pero preservamos lgd y ead ya calculadas
    df_lgd_clean = remove_leakage(df_lgd, strict=True)

    # 5. Guardar datasets procesados en formato Parquet
    print("\n[5/5] Guardando datasets limpios y procesados en data/processed/...")
    pd_parquet_path = os.path.join(processed_dir, "loans_clean_pd.parquet")
    lgd_parquet_path = os.path.join(processed_dir, "loans_clean_lgd.parquet")

    df_pd_clean.to_parquet(pd_parquet_path, index=False, compression="snappy")
    df_lgd_clean.to_parquet(lgd_parquet_path, index=False, compression="snappy")

    size_pd_mb = os.path.getsize(pd_parquet_path) / (1024 * 1024)
    size_lgd_mb = os.path.getsize(lgd_parquet_path) / (1024 * 1024)

    print(f"      Dataset PD guardado:  {pd_parquet_path} ({size_pd_mb:.2f} MB)")
    print(f"      Dataset LGD guardado: {lgd_parquet_path} ({size_lgd_mb:.2f} MB)")

    print("\n" + "="*65)
    print("RESUMEN FINAL DE LA FASE 1a:")
    print("="*65)
    print(f"Dimensión Dataset PD:  {df_pd_clean.shape[0]:,} filas × {df_pd_clean.shape[1]} columnas")
    print(f"Dimensión Dataset LGD: {df_lgd_clean.shape[0]:,} filas × {df_lgd_clean.shape[1]} columnas")
    print(f"Features restantes en PD (sin leakage): {df_pd_clean.shape[1] - 1} predictores + 'target_pd'")
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
