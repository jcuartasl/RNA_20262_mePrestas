"""
src/data/pipeline.py
Módulo para construcción, ajuste (fit) y transformación del pipeline de preprocesamiento.
Usa ColumnTransformer con imputación y estandarización para numéricas,
e imputación y OneHotEncoder con min_frequency para categóricas.
Curso: Redes Neuronales y Algoritmos Bioinspirados - Universidad Nacional de Colombia.
"""

from typing import Tuple, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(
    df_train: pd.DataFrame,
    target_col: str = "target_pd",
    cols_to_exclude: Optional[List[str]] = None,
    save_path: Optional[str] = "models/preprocessor.pkl"
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Construye y ajusta (fit) el pipeline de preprocesamiento exclusivamente sobre df_train.

    Parámetros:
    -----------
    df_train : pd.DataFrame
        Conjunto de datos de entrenamiento (solo Train).
    target_col : str, default='target_pd'
        Nombre de la variable objetivo (no debe ser feature).
    cols_to_exclude : List[str], opcional
        Otras columnas a no incluir como features (ej. issue_d, issue_year, lgd, ead, loan_status).
    save_path : str, opcional
        Ruta donde serializar el preprocesador con joblib.

    Retorna:
    --------
    preprocessor : ColumnTransformer ajustado
    feature_names : List[str] nombres de las columnas transformadas tras OneHotEncoding
    """
    if cols_to_exclude is None:
        cols_to_exclude = []

    # Columnas que nunca deben entrar al preprocesamiento como predictores
    system_excludes = {
        target_col,
        "target_pd",
        "lgd",
        "ead",
        "issue_d",
        "issue_year",
        "loan_status",
    }
    system_excludes.update(cols_to_exclude)

    candidate_cols = [c for c in df_train.columns if c not in system_excludes]

    # Identificar columnas numéricas y categóricas
    numeric_cols = []
    categorical_cols = []

    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df_train[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    print(f"\n[build_preprocessor] Detectadas {len(numeric_cols)} numéricas y {len(categorical_cols)} categóricas.")

    # Sub-pipeline para numéricas: Mediana + StandardScaler
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Sub-pipeline para categóricas: 'missing' + OneHotEncoder (ignorar desconocidas y agrupar raras)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            min_frequency=0.01  # Agrupa categorías con <1% de presencia para reducir dimensionalidad
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Ajustar SOLO sobre el conjunto de entrenamiento
    print("[build_preprocessor] Ajustando preprocesador SOLO sobre conjunto de entrenamiento...")
    preprocessor.fit(df_train)

    # Obtener nombres de las características resultantes
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        # Fallback si get_feature_names_out falla
        feature_names = [f"f_{i}" for i in range(preprocessor.transform(df_train.head(2)).shape[1])]

    print(f"[build_preprocessor] Ajuste completado. Espacio de características final: {len(feature_names)} features.")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(preprocessor, save_path)
        print(f"[build_preprocessor] Preprocesador guardado exitosamente en: {save_path}")

    return preprocessor, feature_names


def transform_split(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    target_col: str,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforma un DataFrame usando el preprocesador previamente ajustado.

    Retorna:
    --------
    X : np.ndarray de float32
    y : np.ndarray de float32
    """
    X = preprocessor.transform(df).astype(np.float32)
    if target_col in df.columns:
        y = df[target_col].to_numpy(dtype=np.float32)
    else:
        y = np.array([], dtype=np.float32)

    return X, y


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "annual_inc": [50000.0, 60000.0, np.nan],
        "home_ownership": ["RENT", "OWN", "RENT"],
        "target_pd": [0, 1, 0],
        "issue_year": [2012, 2013, 2014]
    })
    prep, fnames = build_preprocessor(test_df, target_col="target_pd", save_path="models/test_prep.pkl")
    X, y = transform_split(prep, test_df, "target_pd")
    print("X transformado shape:", X.shape)
    print("Feature names:", fnames)
    if os.path.exists("models/test_prep.pkl"):
        os.remove("models/test_prep.pkl")
