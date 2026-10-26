"""Tablas de comparación justa (guía, sección 8.7) para PD y LGD. Responsable: P4.

Reúne, sin reentrenar las redes, los presupuestos y resultados ya guardados por P2, P3 y P4.
Uso: python -m src.experiments.comparacion_justa
"""
import json
import time

import numpy as np
import pandas as pd

from src.data import load_splits
from src.models.lgd_linear import train_lgd_linear
from src.utils.config import ROOT

T = ROOT / "results" / "tables"


def tabla_pd():
    log = pd.read_csv(T / "pd_logreg_busqueda.csv")
    elegida = log.loc[log["bce"].idxmin()]            # criterio de selección de P3: menor BCE en validación
    busqueda = pd.read_csv(T / "pd_search_real_temporal.csv")
    semillas = json.loads((T / "pd_real_temporal" / "pd_temporal_summary.json").read_text(encoding="utf-8"))
    ablaciones = pd.read_csv(T / "pd_ablations_real" / "pd_temporal_3seeds_summary.csv")
    test = pd.read_csv(T / "p4_pd_eval_test_temporal_resumen.csv").set_index("modelo")
    auc = semillas["auc"]
    filas = [
        {"modelo": "Regresión logística (P3)", "configs_busqueda": len(log), "entrenamientos": len(log),
         "minutos": log["segundos"].sum() / 60, "parametros": int(elegida["n_params"]),
         "n_semillas_val": 1, "auc_val_media": elegida["auc"], "auc_val_sd": np.nan,
         "auc_val_peor": elegida["auc"], "auc_val_mejor": elegida["auc"],
         "interpretabilidad": "alta (coeficientes, puntos exactos)"},
        {"modelo": "MLP (64, 32) ReLU (P2)", "configs_busqueda": len(busqueda),
         "entrenamientos": len(busqueda) + semillas["n_seeds"] + int(ablaciones["n_seeds"].sum()),
         "minutos": (busqueda["time_seconds"].sum() + semillas["total_time_seconds"]
                     + (ablaciones["time_mean_seconds"] * ablaciones["n_seeds"]).sum()) / 60,
         "parametros": semillas["parameters"], "n_semillas_val": semillas["n_seeds"],
         "auc_val_media": auc["mean"], "auc_val_sd": auc["std"], "auc_val_peor": auc["min"], "auc_val_mejor": auc["max"],
         "interpretabilidad": "media (permutación; SHAP pendiente)"},
    ]
    tabla = pd.DataFrame(filas)
    for col, nombre in [("auc", "auc_test"), ("gini", "gini_test"), ("ks", "ks_test"), ("brier", "brier_test")]:
        tabla[nombre] = [test.loc["logreg", f"{col}_media"], test.loc["mlp", f"{col}_media"]]
    tabla["n_semillas_test"] = [int(test.loc["logreg", "n_semillas"]), int(test.loc["mlp", "n_semillas"])]
    return tabla


def tabla_lgd():
    busqueda = pd.read_csv(T / "lgd_busqueda.csv")
    res = pd.read_csv(T / "lgd_semillas_resumen.csv").set_index("modelo")
    test = pd.read_csv(T / "p4_lgd_eval_test_random_resumen.csv").set_index("modelo")
    lin_val = res.loc["lineal (P2, determinista)"]
    red_val = res.loc["mlp bce"]
    d = load_splits("lgd", "random")
    inicio = time.perf_counter()
    train_lgd_linear(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    minutos_lineal = (time.perf_counter() - inicio) / 60
    redes = ["mlp bce", "mlp mse", "mlp beta"]
    filas = [
        {"modelo": "Regresión lineal (P2)", "configs_busqueda": 1, "entrenamientos": 1, "minutos": minutos_lineal,
         "parametros": int(lin_val["n_params"]), "n_semillas_val": 1, "rmse_val_media": lin_val["rmse_media"],
         "rmse_val_sd": np.nan, "rmse_val_mejor": lin_val["rmse_media"], "rmse_val_peor": lin_val["rmse_media"],
         "r2_val": lin_val["r2_media"], "interpretabilidad": "alta (coeficientes)"},
        {"modelo": "MLP BCE [128, 64, 32] tanh (P3)", "configs_busqueda": len(busqueda),
         "entrenamientos": len(busqueda) + int(res.loc[redes, "n_semillas"].sum()),
         "minutos": busqueda["segundos"].sum() / 60 + res.loc[redes, "minutos"].sum(),
         "parametros": int(red_val["n_params"]), "n_semillas_val": int(red_val["n_semillas"]),
         "rmse_val_media": red_val["rmse_media"], "rmse_val_sd": red_val["rmse_sd"],
         "rmse_val_mejor": red_val["rmse_mejor"], "rmse_val_peor": red_val["rmse_peor"], "r2_val": red_val["r2_media"],
         "interpretabilidad": "media (permutación)"},
    ]
    tabla = pd.DataFrame(filas)
    for modelo, i in [("lineal", 0), ("mlp_bce", 1)]:
        for col in ("rmse_media", "rmse_sd", "rmse_min", "rmse_max", "r2_media", "pendiente_media"):
            tabla.loc[i, f"{col}_test"] = test.loc[modelo, col]
        tabla.loc[i, "n_semillas_test"] = int(test.loc[modelo, "n_semillas"])
    return tabla


def main():
    inicio = time.perf_counter()
    for nombre, tabla in [("pd", tabla_pd()), ("lgd", tabla_lgd())]:
        tabla.to_csv(T / f"p4_comparacion_justa_{nombre}.csv", index=False)
        print(f"=== Comparación justa: {nombre.upper()} ===")
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(tabla.set_index("modelo").T.to_string())
        print()
    print(f"[comparacion_justa] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
