"""Contraste descriptivo de las hipótesis H1-H5 de docs/hipotesis.md. Responsable: P4 (con P1).

Uso: python -m src.experiments.contrastar_hipotesis
Muestras: entrenamiento temporal (issue_year <= 2013) y prueba temporal (2015), como en split.py.
"""
import time

import numpy as np
import pandas as pd

from src.interpret.hipotesis import comparar_medias, comparar_tasas, prueba_no_linealidad
from src.utils.config import ROOT

ALFA = 0.05


def _muestras(df):
    return {"train (2007-2013)": df[df["issue_year"] <= 2013], "test (2015)": df[df["issue_year"] == 2015]}


def contrastar(pd_df, lgd_df, col_lgd):
    filas, tramos = [], []
    for muestra, d in _muestras(pd_df).items():
        y = d["target_pd"].to_numpy()
        p75 = d["dti"].quantile(0.75)
        r = comparar_tasas(y[d["dti"] > p75], y[d["dti"] < 15])
        filas.append({"hipotesis": "H1", "muestra": muestra, "grupo_a": f"dti > p75 ({p75:.1f})",
                      "grupo_b": "dti < 15", **r, "criterio": "tasa_a > tasa_b, p < 0.05",
                      "cumple": r["diferencia"] > 0 and r["p_valor"] < ALFA})

        inq = d["inq_last_6mths"]
        r = comparar_tasas(y[inq >= 3], y[inq == 0])
        filas.append({"hipotesis": "H2", "muestra": muestra, "grupo_a": "inq >= 3", "grupo_b": "inq = 0", **r,
                      "criterio": "razon >= 1.5, p < 0.05", "cumple": r["razon"] >= 1.5 and r["p_valor"] < ALFA})

        ru = d["revol_util"]
        umbral = 80 if ru.max() > 2 else 0.8           # porcentaje o fracción
        valido = ru.notna().to_numpy()
        r = comparar_tasas(y[(ru > umbral).to_numpy()], y[(ru <= umbral).to_numpy()])
        nl = prueba_no_linealidad(ru[valido], y[valido])
        filas.append({"hipotesis": "H3", "muestra": muestra, "grupo_a": f"revol_util > {umbral}",
                      "grupo_b": f"revol_util <= {umbral}", **r,
                      "criterio": f"tasa_a > tasa_b, p < 0.05; no linealidad p = {nl['p_valor']:.2g} (LR {nl['lr']:.1f})",
                      "cumple": r["diferencia"] > 0 and r["p_valor"] < ALFA and nl["p_valor"] < ALFA})
        deciles = pd.qcut(ru[valido], 10, duplicates="drop")
        t = pd.DataFrame({"tramo": deciles, "y": y[valido]}).groupby("tramo", observed=True)["y"]
        tramos.append(t.agg(["size", "mean"]).assign(muestra=muestra).reset_index())

        el = d["emp_length"].astype(str)
        if not ((el == "< 1 year").any() and (el == "10+ years").any()):
            raise ValueError(f"categorías de emp_length inesperadas: {sorted(el.unique())}")
        r = comparar_tasas(y[(el == "< 1 year").to_numpy()], y[(el == "10+ years").to_numpy()])
        filas.append({"hipotesis": "H4", "muestra": muestra, "grupo_a": "emp_length < 1 year",
                      "grupo_b": "emp_length 10+ years", **r, "criterio": "tasa_a > tasa_b, p < 0.05",
                      "cumple": r["diferencia"] > 0 and r["p_valor"] < ALFA})

    for muestra, d in _muestras(lgd_df).items():
        r = comparar_medias(d.loc[d["funded_amnt"] > 25000, col_lgd], d.loc[d["funded_amnt"] < 5000, col_lgd])
        filas.append({"hipotesis": "H5", "muestra": muestra, "grupo_a": "funded_amnt > 25 000",
                      "grupo_b": "funded_amnt < 5 000", **r, "criterio": "LGD media_a > media_b, p < 0.05",
                      "cumple": r["diferencia"] > 0 and r["p_valor"] < ALFA})
    return pd.DataFrame(filas), pd.concat(tramos, ignore_index=True)


def main():
    inicio = time.perf_counter()
    pd_df = pd.read_parquet(ROOT / "data" / "processed" / "loans_clean_pd.parquet")
    lgd_df = pd.read_parquet(ROOT / "data" / "processed" / "loans_clean_lgd.parquet")
    col_lgd = next(c for c in ("lgd", "target_lgd") if c in lgd_df.columns)
    tabla, tramos = contrastar(pd_df, lgd_df, col_lgd)
    tabla.to_csv(ROOT / "results" / "tables" / "p4_hipotesis_contraste.csv", index=False)
    tramos.to_csv(ROOT / "results" / "tables" / "p4_hipotesis_revol_util_deciles.csv", index=False)
    cols = ["hipotesis", "muestra", "n_a", "n_b", "valor_a", "valor_b", "diferencia", "razon", "ic_inf", "ic_sup",
            "p_valor", "cumple"]
    with pd.option_context("display.width", 200):
        print(tabla[cols].round(4).to_string(index=False))
        print()
        print(tabla[["hipotesis", "muestra", "criterio"]].to_string(index=False))
        print()
        print(tramos[tramos["muestra"].str.startswith("train")].round(3).to_string(index=False))
    print(f"[contrastar_hipotesis] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
