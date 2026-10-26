"""Predicciones de LGD por semilla con los modelos ya entrenados (sin reentrenar). Responsable: P4.

Usa solo funciones de P3 (cargar_lgd_mlp, predecir_lgd) y de P2 (train_lgd_linear, predict_lgd_linear).
Uso: python -m src.experiments.predict_lgd --split test
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.data import load_splits
from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear
from src.models.lgd_mlp import cargar_lgd_mlp, predecir_lgd
from src.utils.config import ROOT

TABLAS = ROOT / "results" / "tables"
SEMILLAS = range(30)


def main():
    p = argparse.ArgumentParser(description="Predicciones de LGD con los modelos guardados")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--scheme", choices=["random", "temporal"], default="random",
                   help="temporal: solo el lineal; los 30 modelos de red guardados son de la partición aleatoria")
    args = p.parse_args()
    inicio = time.perf_counter()
    datos = load_splits("lgd", args.scheme)

    X = datos[f"X_{args.split}"]
    if args.scheme == "random":
        predecir_red(datos, X, args)
    else:
        print("[predict_lgd] partición temporal: solo el lineal (la red temporal la entrena P3)")

    # Baseline lineal de P2 (determinista): una sola columna
    lineal, _ = train_lgd_linear(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"])
    ruta_lin = TABLAS / f"lgd_pred_{args.split}_{args.scheme}_lineal.csv.gz"
    pd.DataFrame({"seed0": predict_lgd_linear(lineal, X)}).to_csv(ruta_lin, index=False)
    print(f"[predict_lgd] {ruta_lin.name}: ({len(X)}, 1)")
    print(f"[predict_lgd] listo en {time.perf_counter() - inicio:.1f} s")


def predecir_red(datos, X, args):
    # Verificación: cada modelo debe reproducir las predicciones de validación que guardó P3
    referencia = pd.read_csv(TABLAS / "lgd_pred_val_bce.csv.gz")
    modelos = {}
    for s in SEMILLAS:
        modelo, _, perdida = cargar_lgd_mlp(ROOT / "models" / f"lgd_mlp_seed{s}.pt")
        if perdida != "bce":
            raise ValueError(f"seed {s}: se esperaba la red final BCE y el archivo tiene {perdida!r}")
        dif = np.abs(predecir_lgd(modelo, datos["X_val"]) - referencia[f"seed{s}"].to_numpy()).max()
        if dif > 1e-4:
            raise ValueError(f"seed {s}: no reproduce las predicciones de validación de P3 (dif {dif:.2e})")
        modelos[s] = modelo
    print("[predict_lgd] los 30 modelos reproducen la validación de P3")

    # Red final BCE: una columna por semilla
    red = pd.DataFrame({f"seed{s}": predecir_lgd(m, X) for s, m in modelos.items()})
    ruta_red = TABLAS / f"lgd_pred_{args.split}_{args.scheme}_bce.csv.gz"
    red.to_csv(ruta_red, index=False)
    print(f"[predict_lgd] {ruta_red.name}: {red.shape}")


if __name__ == "__main__":
    main()
