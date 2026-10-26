"""Construye la scorecard a partir del modelo de PD. Responsable: P4.

Uso:
    python -m src.experiments.build_scorecard --config configs/scorecard.yaml
    python -m src.experiments.build_scorecard --config configs/scorecard.yaml --override pdo=40 base_score=700
"""
import time

import numpy as np
import pandas as pd

from src.data import load_splits
from src.evaluation.metrics import metricas_pd
from src.models.pd_logreg import cargar_logistica
from src.scorecard.scorecard import Scorecard
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT


def cargar_modelo_pd(cfg):
    """Devuelve (función X -> logit, nombres de variables o None, modelo lineal o None)."""
    tipo = cfg.get("modelo", "logreg")
    if tipo == "logreg":
        modelo, variables = cargar_logistica(ROOT / cfg["ruta_modelo"])
        return modelo.decision_function, list(variables), modelo
    if tipo == "mlp":  # pendiente de probar cuando P2 exporte su modelo final
        import torch
        from src.models.pd_export import load_pd_model
        red = load_pd_model(ROOT / cfg["ruta_modelo"])

        def logit(X):
            with torch.no_grad():
                return red(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().ravel()
        return logit, None, None
    raise ValueError("modelo debe ser 'logreg' o 'mlp'")


def tabla_bandas(y, puntaje, pd_pred, n_bins):
    """Bandas por cuantiles de puntaje (banda 0 = menor puntaje = mayor riesgo)."""
    bordes = np.unique(np.quantile(puntaje, np.linspace(0, 1, n_bins + 1)))
    banda = np.digitize(puntaje, bordes[1:-1], right=True)
    return (pd.DataFrame({"banda": banda, "puntaje": puntaje, "pd": pd_pred, "y": y})
            .groupby("banda")
            .agg(n=("y", "size"), puntaje_min=("puntaje", "min"), puntaje_max=("puntaje", "max"),
                 pd_media=("pd", "mean"), tasa_default=("y", "mean"))
            .reset_index())


def tabla_puntos_logistica(modelo, variables, sc, X_ref):
    """Puntos por variable: -Factor * beta_j * x_j. El intercepto aporta los puntos base."""
    beta = modelo.coef_[0]
    contrib = -sc.factor * beta * np.asarray(X_ref)
    tabla = pd.DataFrame({"variable": variables, "beta": beta, "puntos_por_unidad": -sc.factor * beta,
                          "puntos_medios_abs": np.abs(contrib).mean(axis=0)})
    tabla = tabla.sort_values("puntos_medios_abs", ascending=False)
    base = pd.DataFrame([{"variable": "(puntos base)", "beta": modelo.intercept_[0], "puntos_por_unidad": np.nan,
                          "puntos_medios_abs": sc.offset - sc.factor * modelo.intercept_[0]}])
    return pd.concat([base, tabla], ignore_index=True)


def figura_scorecard(y, s, bandas, ruta, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = np.quantile(s, [0.005, 0.995])
    bins = np.linspace(lo, hi, 50)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    a.hist(s[y == 0], bins=bins, density=True, alpha=0.55, label="buenos (y = 0)")
    a.hist(s[y == 1], bins=bins, density=True, alpha=0.55, label="malos (y = 1)")
    a.set(xlabel="puntaje", ylabel="densidad", title="Distribución de puntajes (0.5 %–99.5 %)")
    a.legend()
    etiquetas = bandas["banda"].astype(str)
    b.bar(etiquetas, bandas["tasa_default"], alpha=0.7, label="tasa observada")
    b.plot(etiquetas, bandas["pd_media"], "ko-", label="PD media predicha")
    b.set(xlabel="banda de puntaje (0 = menor puntaje)", ylabel="tasa de incumplimiento",
          title="Incumplimiento por banda")
    b.legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def run(cfg: dict) -> dict:
    sc = Scorecard.desde_config(cfg)
    logit, variables, modelo = cargar_modelo_pd(cfg)
    d = load_splits("pd", cfg.get("scheme", "temporal"))
    if variables is not None and list(d["feature_names"]) != variables:
        raise ValueError("las variables del modelo no coinciden con las del split")
    nombre = cfg.get("modelo", "logreg")
    tablas, figuras, modelos = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "models"

    s_train = sc.puntaje_desde_logit(logit(d["X_train"]))
    z_test = logit(d["X_test"])
    s_test = sc.puntaje_desde_logit(z_test)
    p_test = 1.0 / (1.0 + np.exp(-z_test))
    y_test = np.asarray(d["y_test"])

    parametros = {"puntaje_base": sc.puntaje_base, "odds_base": sc.odds_base, "pdo": sc.pdo,
                  "factor": sc.factor, "offset": sc.offset}
    pd.DataFrame([parametros]).to_csv(tablas / f"p4_scorecard_parametros_{nombre}.csv", index=False)
    np.savez_compressed(modelos / f"scorecard_referencia_{nombre}.npz",
                        puntajes=np.sort(s_train).astype(np.float32), **parametros)

    bandas = tabla_bandas(y_test, s_test, p_test, int(cfg.get("n_bins", 10)))
    bandas.to_csv(tablas / f"p4_scorecard_bandas_test_{nombre}.csv", index=False)

    salida = {"parametros": parametros, "metricas_test": metricas_pd(y_test, p_test), "bandas": bandas}
    if modelo is not None:
        puntos = tabla_puntos_logistica(modelo, variables, sc, d["X_train"])
        puntos.to_csv(tablas / "scorecard.csv", index=False)
        reconstruido = puntos["puntos_medios_abs"].iloc[0] + (-sc.factor * modelo.coef_[0] * d["X_test"]).sum(axis=1)
        salida["error_aditividad"] = float(np.abs(reconstruido - s_test).max())
        salida["puntos"] = puntos

    figura_scorecard(y_test, s_test, bandas, figuras / f"p4_scorecard_{nombre}_test.png",
                     f"Scorecard ({nombre}, test {cfg.get('scheme', 'temporal')}): "
                     f"base {sc.puntaje_base:g}, odds {sc.odds_base:g}:1, PDO {sc.pdo:g}")
    return salida


def main():
    args = build_parser("Construye el scorecard", "configs/scorecard.yaml").parse_args()
    cfg = init_run(args)
    inicio = time.perf_counter()
    salida = run(cfg)
    print("[build_scorecard] parametros:", {k: round(v, 4) for k, v in salida["parametros"].items()})
    print("[build_scorecard] metricas test:", {k: round(v, 4) for k, v in salida["metricas_test"].items()})
    if "error_aditividad" in salida:
        print("[build_scorecard] max |base + suma de puntos - puntaje|:", salida["error_aditividad"])
        print(salida["puntos"].head(11).round(3).to_string(index=False))
    print(salida["bandas"].round(3).to_string(index=False))
    print(f"[build_scorecard] listo en {time.perf_counter() - inicio:.1f} s")


if __name__ == "__main__":
    main()
