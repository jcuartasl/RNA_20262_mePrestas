"""Calcula la pérdida esperada EL = PD * LGD * EAD. Responsable: P1 (integrado por P5).

Integra scripts/03_expected_loss.py (referencia ingenua) y la combinación con los modelos entrenados.

Se comparan tres variantes sobre el MISMO test temporal (originaciones de `test_year`):
  naive      PD = tasa observada en test, LGD = media de train (<= 2013)         [cota de referencia]
  baselines  regresión logística (PD) + regresión lineal (LGD)                     [modelos de baja complejidad]
  redes      MLP de PD + MLP de LGD                                                [modelos neuronales]
y se contrastan con la pérdida realizada observada (lgd observada * EAD en los incumplidos).

Uso:
    python -m src.experiments.expected_loss --config configs/el.yaml
    python -m src.experiments.expected_loss --override lgd_scheme=random pd_model=logreg
Salidas: results/tables/expected_loss.csv (por crédito), expected_loss_resumen.csv y figuras en results/figures/.
"""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit
from scipy.stats import spearmanr

from src.expected_loss.el import expected_loss, expected_loss_report
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT

TABLAS, FIGURAS = ROOT / "results" / "tables", ROOT / "results" / "figures"


# ------------------------------------------------------------------ datos y alineación
def cargar_cartera(cfg: dict) -> dict:
    """Test temporal: X preprocesada (82), EAD cruda, objetivo y pérdida realizada, todo alineado fila a fila."""
    from src.data import load_splits

    procesados = ROOT / cfg.get("paths", {}).get("processed", "data/processed")
    anio = int(cfg.get("test_year", 2015))
    col_ead = cfg.get("exposure_col", "funded_amnt")

    d = load_splits("pd", "temporal")
    df_pd = pd.read_parquet(procesados / "loans_clean_pd.parquet")
    df_lgd = pd.read_parquet(procesados / "loans_clean_lgd.parquet")

    # df_lgd = incumplidos de df_pd en el mismo orden (compute_lgd filtra y remove_leakage no reordena)
    malos = (df_pd["target_pd"] == 1).to_numpy()
    if malos.sum() != len(df_lgd) or not np.allclose(df_pd.loc[malos, "funded_amnt"].to_numpy(),
                                                     df_lgd["ead"].to_numpy()):
        raise ValueError("loans_clean_pd y loans_clean_lgd no están alineados (¿se regeneró solo uno?)")
    lgd_obs = np.zeros(len(df_pd))
    lgd_obs[malos] = df_lgd["lgd"].to_numpy()
    df_pd = df_pd.assign(lgd_obs=lgd_obs)

    test = df_pd[df_pd["issue_year"] == anio].reset_index(drop=True)
    if len(test) != len(d["X_test"]) or not np.array_equal(test["target_pd"].to_numpy(), d["y_test"].astype(int)):
        raise ValueError(f"el test {anio} de data/processed no coincide con data/splits/pd_temporal_test")
    ead = test[col_ead].to_numpy(dtype=float)
    train_lgd = df_lgd[df_lgd["issue_year"] <= 2013]["lgd"]
    return {"X": d["X_test"], "y": test["target_pd"].to_numpy(), "ead": ead,
            "perdida_real": test["lgd_obs"].to_numpy() * ead, "lgd_media_train": float(train_lgd.mean())}


# ------------------------------------------------------------------ predicciones (importaciones perezosas)
def pd_desde_mlp(ruta, X):
    import torch
    from src.models.pd_export import load_pd_model

    red = load_pd_model(ROOT / ruta)
    with torch.no_grad():
        z = red(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy().ravel()
    return expit(z.astype(float))


def pd_desde_logreg(ruta, X):
    from src.models.pd_logreg import cargar_logistica

    modelo, _ = cargar_logistica(ROOT / ruta)
    return expit(modelo.decision_function(X))


def lgd_desde_mlp(ruta, X):
    from src.models.lgd_mlp import cargar_lgd_mlp, predecir_lgd

    modelo, _, _ = cargar_lgd_mlp(ROOT / ruta)
    return predecir_lgd(modelo, X)


def lgd_desde_lineal(esquema, X):
    from src.data import load_splits
    from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear

    d = load_splits("lgd", esquema)
    modelo, _ = train_lgd_linear(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    return predict_lgd_linear(modelo, X)


def ruta_lgd_mlp(cfg: dict):
    """(ruta, aviso). Prefiere la red temporal; si no existe cae a la aleatoria avisando de la fuga."""
    sem, esquema = int(cfg.get("lgd_seed")), cfg.get("lgd_scheme", "temporal")
    if esquema == "temporal":
        temporal = f"models/lgd_mlp_temporal_seed{sem}.pt"
        if (ROOT / temporal).exists():
            return temporal, ""
        aviso = ("AVISO: no existe " + temporal + "; se usa la red LGD de partición aleatoria, que ya vio "
                 "créditos 2014-2015 al entrenar (EL 2015 optimista). Entrene con: python -m "
                 "src.experiments.train_lgd --override data.scheme=temporal")
        print("[expected_loss]", aviso)
        return f"models/lgd_mlp_seed{sem}.pt", aviso
    return f"models/lgd_mlp_seed{sem}.pt", ""


# ------------------------------------------------------------------ métricas
def resumir(nombre, pd_, lgd, ead, perdida_real, y, aviso=""):
    r = expected_loss(pd_, lgd, ead)
    el, real = r["el_individual"], float(np.sum(perdida_real))
    rho = float(spearmanr(el, perdida_real)[0]) if np.ptp(el) > 0 else float("nan")
    return {"variante": nombre, "n": len(el), "ead_total": r["ead_total"], "el_total": r["el_total"],
            "el_tasa": r["el_rate"], "perdida_real_total": real, "perdida_real_tasa": real / r["ead_total"],
            "el_sobre_real": r["el_total"] / real if real > 0 else float("nan"),
            "mae_credito": float(np.mean(np.abs(el - perdida_real))),
            "rmse_credito": float(np.sqrt(np.mean((el - perdida_real) ** 2))),
            "spearman_el_vs_real": rho, "pd_media": float(np.mean(pd_)), "pd_tasa_obs": float(np.mean(y)),
            "lgd_media": float(np.mean(lgd)), "aviso": aviso}


def figura_comparacion(resumen: pd.DataFrame, ruta):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(resumen["variante"], resumen["el_total"] / 1e6, alpha=0.75, label="EL total")
    ax.axhline(resumen["perdida_real_total"].iloc[0] / 1e6, color="k", ls="--", label="pérdida realizada observada")
    ax.set(ylabel="millones de USD", title="Pérdida esperada vs realizada (test temporal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ experimento
def completar_config(cfg: dict) -> dict:
    """Completa con configs/el.yaml lo que falte (p. ej. cuando la llama demo.py con configs/demo.yaml)."""
    with open(ROOT / "configs" / "el.yaml", encoding="utf-8") as f:
        return {**yaml.safe_load(f), **cfg}


def run(cfg: dict) -> pd.DataFrame:
    t0 = time.perf_counter()
    cfg = completar_config(cfg)
    c = cargar_cartera(cfg)
    n = len(c["y"])
    piso, techo = float(cfg.get("lgd_floor", 0.0)), float(cfg.get("lgd_cap", 1.0))
    limitar = lambda v: np.clip(np.asarray(v, dtype=float), piso, techo)
    print(f"[expected_loss] test {cfg.get('test_year', 2015)}: {n:,} créditos | EAD total ${c['ead'].sum():,.0f}")

    variantes = {"naive": (np.full(n, c["y"].mean()), np.full(n, c["lgd_media_train"]), "")}
    pd_logreg = pd_desde_logreg(cfg.get("pd_logreg_path", "models/pd_logreg.joblib"), c["X"])
    variantes["logreg+lineal"] = (pd_logreg, limitar(lgd_desde_lineal(cfg.get("lgd_scheme", "temporal"), c["X"])), "")
    ruta_red, aviso = ruta_lgd_mlp(cfg)
    pd_red = (pd_desde_mlp(cfg["pd_model_path"], c["X"]) if cfg.get("pd_model", "mlp") == "mlp" else pd_logreg)
    variantes["mlp+mlp"] = (pd_red, limitar(lgd_desde_mlp(ruta_red, c["X"])), aviso)

    filas, detalle = [], pd.DataFrame({"y_true": c["y"], "ead": c["ead"], "perdida_real": c["perdida_real"]})
    for nombre, (p, l, av) in variantes.items():
        filas.append(resumir(nombre, p, l, c["ead"], c["perdida_real"], c["y"], av))
        detalle[f"pd_{nombre}"], detalle[f"lgd_{nombre}"] = p, l
        detalle[f"el_{nombre}"] = expected_loss(p, l, c["ead"])["el_individual"]
    resumen = pd.DataFrame(filas)

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(TABLAS / "expected_loss.csv", index=False)
    resumen.to_csv(TABLAS / "expected_loss_resumen.csv", index=False)
    # figuras: la naive conserva los nombres históricos (distribucion_el.png, el_por_rango_pd.png)
    expected_loss_report(*variantes["naive"][:2], c["ead"], figures_dir=str(FIGURAS))
    expected_loss_report(*variantes["mlp+mlp"][:2], c["ead"], figures_dir=str(FIGURAS), prefix="mlp_")
    figura_comparacion(resumen, FIGURAS / "el_comparacion_modelos.png")

    print("\n" + resumen.drop(columns="aviso").round(4).to_string(index=False))
    print("\n[expected_loss] Nota: las originaciones 2015 no han madurado (sesgo de madurez), por lo que la pérdida "
          "realizada subestima la final; interpretar 'el_sobre_real' con ese sesgo.")
    print(f"[expected_loss] listo en {time.perf_counter() - t0:.1f} s -> results/tables/expected_loss*.csv")
    return resumen


def main():
    args = build_parser("Calcula la pérdida esperada", "configs/el.yaml").parse_args()
    cfg = init_run(args)
    run(cfg)


if __name__ == "__main__":
    main()
