"""Importancia por permutación del modelo de PD. Responsable: P4.

Uso:
    python -m src.experiments.interpret_pd --config configs/interpret_pd.yaml
    python -m src.experiments.interpret_pd --config configs/interpret_pd.yaml --override modelo=mlp ruta_modelo=models/pd_mlp.pt
"""
import time

import joblib

from src.data import load_splits
from src.experiments.build_scorecard import cargar_modelo_pd
from src.interpret.permutation import figura_importancia, grupos_por_variable, importancia_permutacion
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT


def run(cfg):
    logit, _, _ = cargar_modelo_pd(cfg)
    d = load_splits("pd", cfg["scheme"])
    pre = joblib.load(ROOT / "models" / "preprocessor.pkl")
    originales = [c for nombre, _, cols in pre.transformers_ if nombre != "remainder" for c in cols]
    grupos = grupos_por_variable(list(d["feature_names"]), originales, cfg.get("grupos_extra"))
    tabla = importancia_permutacion(logit, d[f"X_{cfg['split']}"], d[f"y_{cfg['split']}"], grupos,
                                    n_repeticiones=int(cfg["n_repeticiones"]), seed=int(cfg["seed"]))
    sufijo = f"{cfg['modelo']}_{cfg['split']}_{cfg['scheme']}"
    tabla.to_csv(ROOT / "results" / "tables" / f"p4_importancia_pd_{sufijo}.csv", index=False)
    figura_importancia(tabla, ROOT / "results" / "figures" / f"p4_importancia_pd_{sufijo}.png",
                       f"Importancia por permutación ({cfg['modelo']}, {cfg['split']} {cfg['scheme']})")
    return tabla


def main():
    args = build_parser("Importancia por permutación de PD", "configs/interpret_pd.yaml").parse_args()
    cfg = init_run(args)
    inicio = time.perf_counter()
    tabla = run(cfg)
    print(tabla.head(15).round(4).to_string(index=False))
    print(f"[interpret_pd] AUC base {tabla['metrica_base'].iloc[0]:.4f} | listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
