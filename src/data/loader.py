"""
src/data/loader.py
Módulo de Contrato de Interfaz para el equipo (P1 produce, P2, P3 y P4 consumen).
Carga y retorna las particiones de entrenamiento, validación y prueba preprocesadas.
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Dict, Any, List
import os
import pandas as pd
import numpy as np


def load_splits(task: str = "pd", scheme: str = "temporal") -> Dict[str, Any]:
    """
    Contrato oficial de carga de datos para modelado.

    Parámetros:
    -----------
    task : str, default='pd'
        'pd'  : Probabilidad de Incumplimiento (clasificación binaria: target_pd in {0, 1})
        'lgd' : Pérdida Dado el Incumplimiento (regresión continua: lgd in [0, 1])
    scheme : str, default='temporal'
        'temporal' : Partición cronológica por issue_year (Train <= 2013, Val = 2014, Test = 2015)
        'random'   : Partición aleatoria estratificada (70% Train, 15% Val, 15% Test)

    Retorna:
    --------
    dict con las siguientes claves:
        - 'X_train' : np.ndarray de float32
        - 'y_train' : np.ndarray de float32
        - 'X_val'   : np.ndarray de float32
        - 'y_val'   : np.ndarray de float32
        - 'X_test'  : np.ndarray de float32
        - 'y_test'  : np.ndarray de float32
        - 'feature_names' : List[str] nombres de las columnas transformadas
    """
    task = task.lower().strip()
    scheme = scheme.lower().strip()

    if task not in ["pd", "lgd"]:
        raise ValueError(f"Task '{task}' no válida. Debe ser 'pd' o 'lgd'.")
    if scheme not in ["temporal", "random"]:
        raise ValueError(f"Scheme '{scheme}' no válido. Debe ser 'temporal' o 'random'.")

    # Localizar directorio base del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    splits_dir = os.path.join(project_root, "data", "splits")

    # Nombres de archivos parquet esperados
    train_file = os.path.join(splits_dir, f"{task}_{scheme}_train.parquet")
    val_file   = os.path.join(splits_dir, f"{task}_{scheme}_val.parquet")
    test_file  = os.path.join(splits_dir, f"{task}_{scheme}_test.parquet")

    for fpath in [train_file, val_file, test_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"No se encontró el archivo de partición requerido: {fpath}\n"
                f"Asegúrese de haber ejecutado 'python scripts/02_split_and_preprocess.py'."
            )

    target_name = "target_pd" if task == "pd" else "lgd"

    # Cargar archivos parquet
    df_train = pd.read_parquet(train_file)
    df_val   = pd.read_parquet(val_file)
    df_test  = pd.read_parquet(test_file)

    # Identificar columnas de características (todas excepto target_name)
    feature_cols = [c for c in df_train.columns if c != target_name]

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_train = df_train[target_name].to_numpy(dtype=np.float32)

    X_val   = df_val[feature_cols].to_numpy(dtype=np.float32)
    y_val   = df_val[target_name].to_numpy(dtype=np.float32)

    X_test  = df_test[feature_cols].to_numpy(dtype=np.float32)
    y_test  = df_test[target_name].to_numpy(dtype=np.float32)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_cols,
    }


if __name__ == "__main__":
    print("Módulo loader.py listo. Ejecutando verificación rápida...")
    try:
        data = load_splits("pd", "temporal")
        print("Carga exitosa! Dimensiones:")
        print(f"X_train: {data['X_train'].shape}, y_train: {data['y_train'].shape}")
        print(f"X_val:   {data['X_val'].shape}, y_val:   {data['y_val'].shape}")
        print(f"X_test:  {data['X_test'].shape}, y_test:  {data['y_test'].shape}")
        print(f"Número de features: {len(data['feature_names'])}")
    except Exception as e:
        print(f"Aviso informativo: {e}")
