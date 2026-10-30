"""
src/data/split.py
Módulo para partición temporal (por fecha de originación issue_d / issue_year)
y partición aleatoria estratificada.
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def temporal_split(
    df: pd.DataFrame,
    target_col: str = "target_pd"
) -> Dict[str, pd.DataFrame]:
    """
    Realiza la partición temporal cronológica del dataset según issue_year:
    - Train: 2007 a 2013 (originaciones antiguas / maduras)
    - Val:   2014 (originaciones intermedias)
    - Test:  2015 (originaciones recientes, documentando sesgo de madurez)

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame limpio que contiene 'issue_year' y target_col.
    target_col : str, default='target_pd'
        Nombre de la variable objetivo para reportar tasas.

    Retorna:
    --------
    dict con DataFrames: {'train': df_train, 'val': df_val, 'test': df_test}
    """
    if "issue_year" not in df.columns:
        raise KeyError("La columna 'issue_year' no está presente en el DataFrame.")

    train_mask = (df["issue_year"] <= 2013)
    val_mask = (df["issue_year"] == 2014)
    test_mask = (df["issue_year"] == 2015)

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    total_len = len(df)
    print("\n--- Partición Temporal (por issue_year) ---")
    for name, split_df in [("Train (2007-2013)", df_train), ("Val (2014)", df_val), ("Test (2015)", df_test)]:
        pct = len(split_df) / total_len * 100 if total_len > 0 else 0
        if target_col in split_df.columns:
            target_mean = split_df[target_col].mean()
            print(f"  {name:<18}: {len(split_df):>7,} filas ({pct:>5.1f}%) | {target_col} media = {target_mean:.4f}")
        else:
            print(f"  {name:<18}: {len(split_df):>7,} filas ({pct:>5.1f}%)")

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }


def random_split(
    df: pd.DataFrame,
    target_col: str = "target_pd",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 90
) -> Dict[str, pd.DataFrame]:
    """
    Realiza una partición aleatoria estratificada (70% Train, 15% Val, 15% Test)
    como línea base comparativa frente a la partición temporal.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame limpio.
    target_col : str, default='target_pd'
        Columna para estratificación. Si es continua o tiene un solo valor, no se estratifica.
    train_size, val_size, test_size : float
        Proporciones relativas (deben sumar 1.0).
    random_state : int
        Semilla reproducible.

    Retorna:
    --------
    dict con DataFrames: {'train': df_train, 'val': df_val, 'test': df_test}
    """
    total_len = len(df)
    # Comprobar si target_col es binario discreto para estratificar
    stratify_col = None
    if target_col in df.columns:
        unique_vals = df[target_col].dropna().unique()
        if len(unique_vals) <= 10:  # Clasificación / binaria
            stratify_col = df[target_col]

    # Paso 1: separar Train (70%) y Temp (30%)
    temp_size = val_size + test_size
    df_train, df_temp = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_col
    )

    # Paso 2: separar Temp en Val (50% de 30% = 15%) y Test (50% de 30% = 15%)
    val_ratio = val_size / temp_size
    stratify_temp = None
    if stratify_col is not None:
        stratify_temp = df_temp[target_col]

    df_val, df_test = train_test_split(
        df_temp,
        test_size=(1.0 - val_ratio),
        random_state=random_state,
        stratify=stratify_temp
    )

    print("\n--- Partición Aleatoria Estratificada ---")
    for name, split_df in [("Train (70%)", df_train), ("Val (15%)", df_val), ("Test (15%)", df_test)]:
        pct = len(split_df) / total_len * 100 if total_len > 0 else 0
        if target_col in split_df.columns:
            target_mean = split_df[target_col].mean()
            print(f"  {name:<18}: {len(split_df):>7,} filas ({pct:>5.1f}%) | {target_col} media = {target_mean:.4f}")
        else:
            print(f"  {name:<18}: {len(split_df):>7,} filas ({pct:>5.1f}%)")

    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }


if __name__ == "__main__":
    dummy = pd.DataFrame({
        "issue_year": [2010, 2012, 2014, 2015, 2015, 2013],
        "target_pd": [0, 1, 0, 1, 0, 1]
    })
    print("Prueba temporal:")
    ts = temporal_split(dummy)
    print("Prueba aleatoria:")
    rs = random_split(dummy)
