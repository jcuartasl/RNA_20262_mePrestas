"""Importancia por permutación de LGD: aumento de RMSE al permutar cada variable. Responsable: P4.

Uso: python -m src.experiments.interpret_lgd --config configs/interpret_lgd.yaml
"""
import time

import joblib

from src.data import load_splits
from src.evaluation.metrics import rmse
from src.interpret.permutation import figura_importancia, grupos_por_variable, importancia_permutacion
from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear
from src.models.lgd_mlp import cargar_lgd_mlp, predecir_lgd
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT


def menos_rmse(y, p):
    """Mayor es mejor, para que la importancia (base - permutada) sea el aumento de RMSE."""
    return -rmse(y, p)


def predictores(cfg, d):
    s = int(cfg["semilla_red"])
    red, _, perdida = cargar_lgd_mlp(ROOT / "models" / f"lgd_mlp_seed{s}.pt")
    salida = {f"mlp_{perdida}_seed{s}": lambda X: predecir_lgd(red, X)}
    if cfg.get("incluir_lineal", True):
        lineal, _ = train_lgd_linear(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
        salida["lineal"] = lambda X: predict_lgd_linear(lineal, X)
    return salida


def run(cfg):
    d = load_splits("lgd", cfg["scheme"])
    pre = joblib.load(ROOT / "models" / "preprocessor.pkl")
    originales = [c for nombre, _, cols in pre.transformers_ if nombre != "remainder" for c in cols]
    grupos = grupos_por_variable(list(d["feature_names"]), originales, cfg.get("grupos_extra"))
    X, y = d[f"X_{cfg['split']}"], d[f"y_{cfg['split']}"]
    tablas = {}
    for nombre, predecir in predictores(cfg, d).items():
        tabla = importancia_permutacion(predecir, X, y, grupos, metrica=menos_rmse,
                                        n_repeticiones=int(cfg["n_repeticiones"]), seed=int(cfg["seed"]))
        tabla = tabla.rename(columns={"metrica_base": "rmse_base"}).assign(rmse_base=lambda t: -t["rmse_base"])
        sufijo = f"{nombre}_{cfg['split']}_{cfg['scheme']}"
        tabla.to_csv(ROOT / "results" / "tables" / f"p4_importancia_lgd_{sufijo}.csv", index=False)
        figura_importancia(tabla, ROOT / "results" / "figures" / f"p4_importancia_lgd_{sufijo}.png",
                           f"Importancia por permutación de LGD ({nombre}, {cfg['split']} {cfg['scheme']})",
                           etiqueta="aumento de RMSE al permutar (media ± sd)")
        tablas[nombre] = tabla
    return tablas


def main():
    args = build_parser("Importancia por permutación de LGD", "configs/interpret_lgd.yaml").parse_args()
    cfg = init_run(args)
    inicio = time.perf_counter()
    for nombre, tabla in run(cfg).items():
        print(f"=== {nombre} (RMSE base {tabla['rmse_base'].iloc[0]:.4f}) ===")
        print(tabla.drop(columns="rmse_base").head(12).round(4).to_string(index=False))
    print(f"[interpret_lgd] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
