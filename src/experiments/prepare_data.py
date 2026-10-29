"""Limpia los datos, define los objetivos PD/LGD y genera los splits. Responsable: P1 (integrado por P5).

Uso:
    python -m src.experiments.prepare_data --config configs/default.yaml
    python -m src.experiments.prepare_data --override prepare.clean=true     # exige data/raw/loan.csv
    python -m src.experiments.prepare_data --override prepare.clean=false    # solo re-particiona lo ya procesado

Comportamiento por defecto (prepare.clean = auto):
  * si existe data/raw/loan.csv -> fase 1a (limpieza, LGD, fuga) + fase 1b (splits y preprocesador);
  * si NO existe (el repo no lo trae: pesa > 100 MB) -> salta la 1a y usa data/processed/*.parquet.
En modo demo (sample_fraction < 1) NO se reescribe nada: solo se valida el contrato de load_splits(),
para no pisar models/preprocessor.pkl ni data/splits/ que usa la app.
"""
import time

import numpy as np
import pandas as pd

from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT

ESQUEMAS = [("pd", "temporal"), ("pd", "random"), ("lgd", "temporal"), ("lgd", "random")]
OBJETIVO = {"pd": "target_pd", "lgd": "lgd"}


def _ruta(cfg: dict, clave: str, defecto: str):
    return ROOT / cfg.get("paths", {}).get(clave, defecto)


# ---------------------------------------------------------------- fase 1a
def limpiar_datos(cfg: dict) -> dict:
    """loan.csv -> target_pd, LGD/EAD, auditoría de fuga -> data/processed/*.parquet."""
    from src.data.clean import clean_loans
    from src.data.leakage import generate_leakage_report, remove_leakage
    from src.data.lgd import compute_lgd

    raw = _ruta(cfg, "raw", "data/raw") / "loan.csv"
    procesados = _ruta(cfg, "processed", "data/processed")
    tablas, figuras = ROOT / "results" / "tables", ROOT / "results" / "figures"
    for carpeta in (procesados, tablas, figuras):
        carpeta.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        raise FileNotFoundError(f"No se encontró el archivo original: {raw}")

    t0 = time.perf_counter()
    print(f"[prepare_data] 1a/ cargando {raw} ...")
    df_raw = pd.read_csv(raw, low_memory=False)
    print(f"[prepare_data]      {len(df_raw):,} filas x {df_raw.shape[1]} columnas ({time.perf_counter() - t0:.1f} s)")

    df_clean, rep = clean_loans(df_raw, null_threshold=cfg.get("prepare", {}).get("null_threshold", 0.50))
    print(f"[prepare_data]      resueltos: {rep['n_rows_resolved']:,} | malos: {rep['n_defaults']:,} | "
          f"tasa de default: {rep['default_rate'] * 100:.2f} %")

    # LGD y EAD se construyen ANTES de quitar las columnas de recuperación (son fuga para PD)
    df_lgd, est = compute_lgd(df_clean, plot_histogram=True, figure_path=str(figuras / "distribucion_lgd.png"))
    print(f"[prepare_data]      LGD: n={est['n_defaults']:,} media={est['mean']:.4f} "
          f"ceros={est['pct_exact_zero']:.2f} % unos={est['pct_exact_one']:.2f} %")

    generate_leakage_report(df_clean, output_path=str(tablas / "leakage_report.csv"))
    df_pd_clean = remove_leakage(df_clean, strict=True)
    df_lgd_clean = remove_leakage(df_lgd, strict=True)   # conserva lgd y ead

    df_pd_clean.to_parquet(procesados / "loans_clean_pd.parquet", index=False, compression="snappy")
    df_lgd_clean.to_parquet(procesados / "loans_clean_lgd.parquet", index=False, compression="snappy")
    print(f"[prepare_data]      PD {df_pd_clean.shape} | LGD {df_lgd_clean.shape} -> {procesados}")
    return {"n_pd": len(df_pd_clean), "n_lgd": len(df_lgd_clean), "default_rate": rep["default_rate"]}


# ---------------------------------------------------------------- fase 1b
def _guardar(X, y, nombres, objetivo, ruta):
    salida = pd.DataFrame(X, columns=nombres)
    salida[objetivo] = y
    salida.to_parquet(ruta, index=False, compression="snappy")


def particionar(cfg: dict) -> dict:
    """data/processed -> particiones temporal y aleatoria + preprocesador (ajustado solo con train temporal de PD)."""
    from src.data.pipeline import build_preprocessor, transform_split
    from src.data.split import random_split, temporal_split

    procesados, splits = _ruta(cfg, "processed", "data/processed"), _ruta(cfg, "splits", "data/splits")
    modelos = ROOT / "models"
    splits.mkdir(parents=True, exist_ok=True)
    modelos.mkdir(parents=True, exist_ok=True)

    df_pd = pd.read_parquet(procesados / "loans_clean_pd.parquet")
    df_lgd = pd.read_parquet(procesados / "loans_clean_lgd.parquet")
    print(f"[prepare_data] 1b/ PD {df_pd.shape} | LGD {df_lgd.shape}")

    # La semilla de las particiones aleatorias es una sola y está documentada (default.yaml: split_seed)
    semilla = int(cfg["split_seed"])
    proporciones = cfg.get("split", {"train": 0.7, "val": 0.15, "test": 0.15})
    kw = dict(train_size=proporciones["train"], val_size=proporciones["val"],
              test_size=proporciones["test"], random_state=semilla)
    particiones = {
        ("pd", "temporal"): temporal_split(df_pd, target_col="target_pd"),
        ("pd", "random"): random_split(df_pd, target_col="target_pd", **kw),
        ("lgd", "temporal"): temporal_split(df_lgd, target_col="lgd"),
        ("lgd", "random"): random_split(df_lgd, target_col="lgd", **kw),
    }

    # Un único preprocesador (train temporal de PD) para PD y LGD: regla de comparación justa
    preprocesador, nombres = build_preprocessor(
        df_train=particiones[("pd", "temporal")]["train"], target_col="target_pd",
        cols_to_exclude=["lgd", "ead", "issue_d", "issue_year", "loan_status"],
        save_path=str(modelos / "preprocessor.pkl"))

    for (tarea, esquema), partes in particiones.items():
        for nombre, df_sub in partes.items():
            X, y = transform_split(preprocesador, df_sub, target_col=OBJETIVO[tarea])
            ruta = splits / f"{tarea}_{esquema}_{nombre}.parquet"
            _guardar(X, y, nombres, OBJETIVO[tarea], ruta)
            print(f"[prepare_data]      {ruta.name}: {X.shape[0]:,} filas")
    return {"n_features": len(nombres), "split_seed": semilla}


# ---------------------------------------------------------------- validación
def validar_contrato() -> dict:
    """load_splits() debe funcionar en los 4 esquemas, con las mismas columnas y sin NaN."""
    from src.data import load_splits

    columnas, formas = None, {}
    for tarea, esquema in ESQUEMAS:
        d = load_splits(tarea, esquema)
        if columnas is None:
            columnas = d["feature_names"]
        elif d["feature_names"] != columnas:
            raise ValueError(f"{tarea}/{esquema}: las columnas difieren de las de PD temporal")
        for k in ("X_train", "X_val", "X_test"):
            if not np.isfinite(d[k]).all():
                raise ValueError(f"{tarea}/{esquema}/{k}: contiene NaN o infinitos")
        formas[f"{tarea}_{esquema}"] = tuple(d["X_train"].shape)
        print(f"[prepare_data] contrato ok: load_splits('{tarea}', '{esquema}') X_train={d['X_train'].shape}")
    return formas


def run(cfg: dict) -> dict:
    """Contrato (demo.py): run(cfg) -> dict. No lanza NotImplementedError."""
    t0 = time.perf_counter()
    prep = cfg.get("prepare", {})
    demo = float(cfg.get("sample_fraction", 1.0)) < 1.0
    if demo and not prep.get("force", False):
        print("[prepare_data] modo demo: no se reescriben datos ni preprocesador; solo se valida el contrato")
        return {"modo": "demo", "formas": validar_contrato(), "segundos": time.perf_counter() - t0}

    modo = str(prep.get("clean", "auto")).lower()
    existe_raw = (_ruta(cfg, "raw", "data/raw") / "loan.csv").exists()
    salida = {"modo": "completo"}
    if modo in ("true", "1") or (modo == "auto" and existe_raw):
        salida.update(limpiar_datos(cfg))
    elif modo == "auto":
        print("[prepare_data] no hay data/raw/loan.csv: se omite la fase 1a y se usa data/processed/ (normal en un clon)")
    salida.update(particionar(cfg))
    salida["formas"] = validar_contrato()
    salida["segundos"] = time.perf_counter() - t0
    print(f"[prepare_data] listo en {salida['segundos']:.1f} s")
    return salida


def main():
    args = build_parser("Prepara datos y splits", "configs/default.yaml").parse_args()
    cfg = init_run(args)
    print("[prepare_data] config:", cfg)
    run(cfg)


if __name__ == "__main__":
    main()
