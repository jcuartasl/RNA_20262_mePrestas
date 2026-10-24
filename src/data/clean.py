"""
src/data/clean.py
Módulo para limpieza y codificación del objetivo binario de Probabilidad de Incumplimiento (PD).
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np


def get_pd_target_mapping() -> Dict[str, Any]:
    """
    Retorna el diccionario de mapeo oficial de loan_status a PD según la guía del trabajo:
    - 0 (Buen pagador confirmado): Fully Paid, Does not meet the credit policy. Status:Fully Paid
    - 1 (Mal pagador / Default): Charged Off, Default, Late (31-120 days), Does not meet the credit policy. Status:Charged Off
    - NA (Excluidos de entrenamiento por no tener desenlace observado): Current, Issued, In Grace Period, Late (16-30 days)
    """
    return {
        "Fully Paid": 0,
        "Does not meet the credit policy. Status:Fully Paid": 0,
        "Charged Off": 1,
        "Default": 1,
        "Late (31-120 days)": 1,
        "Does not meet the credit policy. Status:Charged Off": 1,
        "Current": np.nan,
        "Issued": np.nan,
        "In Grace Period": np.nan,
        "Late (16-30 days)": np.nan,
    }


def clean_loans(
    df: pd.DataFrame,
    null_threshold: float = 0.50,
    text_cols_to_drop: List[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Realiza la limpieza inicial de los créditos y codifica la variable objetivo binaria target_pd.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame cargado desde loan.csv original.
    null_threshold : float, default=0.50
        Umbral máximo de valores nulos permitido (columnas con > 50% se descartan).
    text_cols_to_drop : List[str], opcional
        Lista de identificadores y columnas de texto libre a eliminar.

    Retorna:
    --------
    cleaned_df : pd.DataFrame
        DataFrame limpio con la columna 'target_pd' (int64) e 'issue_year' (int64).
    report : dict
        Métricas de resumen del proceso de limpieza.
    """
    if text_cols_to_drop is None:
        text_cols_to_drop = ["id", "member_id", "url", "desc", "title"]

    n_rows_initial = len(df)
    n_cols_initial = df.shape[1]

    # 1. Mapeo del objetivo binario PD a partir de loan_status
    mapping = get_pd_target_mapping()
    target_series = df["loan_status"].map(mapping)

    # Filtrar solo créditos con estado resuelto (no NA)
    valid_mask = target_series.notna()
    df_resolved = df[valid_mask].copy()
    df_resolved["target_pd"] = target_series[valid_mask].astype(int)

    n_rows_resolved = len(df_resolved)
    n_defaults = int(df_resolved["target_pd"].sum())
    default_rate = float(df_resolved["target_pd"].mean())

    # 2. Parseo de issue_d y extracción de año y mes
    if "issue_d" in df_resolved.columns:
        issue_d_parsed = pd.to_datetime(df_resolved["issue_d"], format="%b-%Y", errors="coerce")
        df_resolved["issue_year"] = issue_d_parsed.dt.year.astype("Int64")
    else:
        df_resolved["issue_year"] = np.nan

    # 3. Eliminar columnas con porcentaje de nulos mayor al umbral (calculado sobre el conjunto resuelto)
    null_pct = df_resolved.isnull().mean()
    high_null_cols = null_pct[null_pct > null_threshold].index.tolist()
    df_resolved = df_resolved.drop(columns=high_null_cols, errors="ignore")

    # 4. Eliminar identificadores y texto libre especificados
    dropped_text_cols = [c for c in text_cols_to_drop if c in df_resolved.columns]
    df_resolved = df_resolved.drop(columns=dropped_text_cols, errors="ignore")

    report = {
        "n_rows_initial": n_rows_initial,
        "n_rows_resolved": n_rows_resolved,
        "n_rows_excluded": n_rows_initial - n_rows_resolved,
        "n_defaults": n_defaults,
        "n_non_defaults": n_rows_resolved - n_defaults,
        "default_rate": default_rate,
        "n_cols_initial": n_cols_initial,
        "n_cols_final": df_resolved.shape[1],
        "high_null_cols_dropped": high_null_cols,
        "text_cols_dropped": dropped_text_cols,
    }

    return df_resolved, report


if __name__ == "__main__":
    import os
    raw_path = "data/raw/loan.csv"
    if os.path.exists(raw_path):
        print(f"Cargando muestra de prueba desde {raw_path}...")
        sample_df = pd.read_csv(raw_path, nrows=5000, low_memory=False)
        cleaned_sample, rep = clean_loans(sample_df)
        print("Limpieza de prueba completada exitosamente:")
        for k, v in rep.items():
            if isinstance(v, list):
                print(f"  {k}: {len(v)} columnas -> {v[:5]}...")
            else:
                print(f"  {k}: {v}")
    else:
        print(f"No se encontró {raw_path}. Módulo listo para ser importado.")
