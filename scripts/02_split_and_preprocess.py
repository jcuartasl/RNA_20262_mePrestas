"""
scripts/02_split_and_preprocess.py
Script de ejecución completa para Fase 1b:
1. Carga data/processed/loans_clean_pd.parquet y loans_clean_lgd.parquet
2. Genera particiones temporales (por issue_year) y aleatorias estratificadas (70/15/15)
3. Construye y ajusta el preprocesador (ColumnTransformer) SOLO sobre Train temporal de PD
4. Guarda el objeto en models/preprocessor.pkl
5. Transforma todas las particiones (temporal y random) para PD y para LGD
6. Guarda los datasets transformados en data/splits/*.parquet
7. Valida el funcionamiento del contrato load_splits()

Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.split import temporal_split, random_split
from src.data.pipeline import build_preprocessor, transform_split
from src.data.loader import load_splits


def save_split_parquet(X: np.ndarray, y: np.ndarray, feature_names: list, target_name: str, file_path: str):
    """Auxiliar para guardar una partición procesada como DataFrame en formato Parquet."""
    df_out = pd.DataFrame(X, columns=feature_names)
    df_out[target_name] = y
    df_out.to_parquet(file_path, index=False, compression="snappy")


def main():
    start_time = time.time()
    processed_dir = os.path.join(project_root, "data", "processed")
    splits_dir = os.path.join(project_root, "data", "splits")
    models_dir = os.path.join(project_root, "models")

    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    pd_path = os.path.join(processed_dir, "loans_clean_pd.parquet")
    lgd_path = os.path.join(processed_dir, "loans_clean_lgd.parquet")

    print("=================================================================")
    print("FASE 1b: PIPELINE DE PREPROCESAMIENTO, PARTICIONES Y CONTRATOS")
    print("=================================================================")

    # 1. Cargar datasets limpios
    print(f"\n[1/6] Cargando datos limpios...")
    df_pd = pd.read_parquet(pd_path)
    df_lgd = pd.read_parquet(lgd_path)
    print(f"      Dataset PD cargado:  {df_pd.shape[0]:,} filas × {df_pd.shape[1]} columnas")
    print(f"      Dataset LGD cargado: {df_lgd.shape[0]:,} filas × {df_lgd.shape[1]} columnas")

    # 2. Generar particiones temporales y aleatorias
    print(f"\n[2/6] Generando particiones temporales y aleatorias para PD...")
    splits_pd_temporal = temporal_split(df_pd, target_col="target_pd")
    splits_pd_random = random_split(df_pd, target_col="target_pd", random_state=42)

    print(f"\n[3/6] Generando particiones temporales y aleatorias para LGD...")
    splits_lgd_temporal = temporal_split(df_lgd, target_col="lgd")
    splits_lgd_random = random_split(df_lgd, target_col="lgd", random_state=42)

    # 3. Construir y ajustar el preprocesador oficial SOLO sobre el Train temporal de PD
    print(f"\n[4/6] Construyendo preprocesador oficial sobre Train temporal de PD...")
    preprocessor_pkl_path = os.path.join(models_dir, "preprocessor.pkl")
    preprocessor, feature_names = build_preprocessor(
        df_train=splits_pd_temporal["train"],
        target_col="target_pd",
        cols_to_exclude=["lgd", "ead", "issue_d", "issue_year", "loan_status"],
        save_path=preprocessor_pkl_path
    )

    # 4. Transformar y guardar particiones de PD
    print(f"\n[5/6] Transformando y guardando particiones en {splits_dir}...")
    
    # PD Temporal
    for split_name in ["train", "val", "test"]:
        df_sub = splits_pd_temporal[split_name]
        X_sub, y_sub = transform_split(preprocessor, df_sub, target_col="target_pd")
        out_path = os.path.join(splits_dir, f"pd_temporal_{split_name}.parquet")
        save_split_parquet(X_sub, y_sub, feature_names, "target_pd", out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"      Guardado: pd_temporal_{split_name}.parquet ({X_sub.shape[0]:,} filas, {size_mb:.2f} MB)")

    # PD Random
    for split_name in ["train", "val", "test"]:
        df_sub = splits_pd_random[split_name]
        X_sub, y_sub = transform_split(preprocessor, df_sub, target_col="target_pd")
        out_path = os.path.join(splits_dir, f"pd_random_{split_name}.parquet")
        save_split_parquet(X_sub, y_sub, feature_names, "target_pd", out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"      Guardado: pd_random_{split_name}.parquet ({X_sub.shape[0]:,} filas, {size_mb:.2f} MB)")

    # LGD Temporal (mismo preprocesador para garantizar regla de comparación justa)
    for split_name in ["train", "val", "test"]:
        df_sub = splits_lgd_temporal[split_name]
        X_sub, y_sub = transform_split(preprocessor, df_sub, target_col="lgd")
        out_path = os.path.join(splits_dir, f"lgd_temporal_{split_name}.parquet")
        save_split_parquet(X_sub, y_sub, feature_names, "lgd", out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"      Guardado: lgd_temporal_{split_name}.parquet ({X_sub.shape[0]:,} filas, {size_mb:.2f} MB)")

    # LGD Random
    for split_name in ["train", "val", "test"]:
        df_sub = splits_lgd_random[split_name]
        X_sub, y_sub = transform_split(preprocessor, df_sub, target_col="lgd")
        out_path = os.path.join(splits_dir, f"lgd_random_{split_name}.parquet")
        save_split_parquet(X_sub, y_sub, feature_names, "lgd", out_path)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"      Guardado: lgd_random_{split_name}.parquet ({X_sub.shape[0]:,} filas, {size_mb:.2f} MB)")

    # 5. Validación del contrato de interfaz con load_splits()
    print(f"\n[6/6] Verificando contrato de interfaz load_splits()...")
    data_pd_temp = load_splits("pd", "temporal")
    data_lgd_temp = load_splits("lgd", "temporal")
    data_pd_rand = load_splits("pd", "random")

    print(f"      load_splits('pd', 'temporal') -> X_train: {data_pd_temp['X_train'].shape}, y_train: {data_pd_temp['y_train'].shape}")
    print(f"      load_splits('lgd', 'temporal') -> X_train: {data_lgd_temp['X_train'].shape}, y_train: {data_lgd_temp['y_train'].shape}")
    print(f"      load_splits('pd', 'random')    -> X_train: {data_pd_rand['X_train'].shape}, y_train: {data_pd_rand['y_train'].shape}")

    print("\n" + "="*65)
    print("RESUMEN FINAL DE LA FASE 1b:")
    print("="*65)
    print(f"Espacio de características transformadas: {len(feature_names)} variables.")
    print(f"Modelos (P2, P3, P4) pueden consumir directamente load_splits(task, scheme).")
    print(f"Preprocesador para app (P5): {preprocessor_pkl_path}")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
