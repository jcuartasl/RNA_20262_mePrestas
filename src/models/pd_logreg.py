"""Regresión logística de referencia para PD con búsqueda de regularización L1/L2."""
import time

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.data import load_splits
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT
from src.utils.registry import LOG_PATH, log_run


def entrenar_logistica(X, y, C, l1_ratio, seed=90, max_iter=1000, tol=1e-4):
    """Ajusta la logística por máxima verosimilitud con penalización l2 (l1_ratio=0) o l1 (l1_ratio=1)."""
    # lbfgs solo admite l2; saga hace pasos proximales y por eso la l1 deja coeficientes en 0 exacto
    solver = "lbfgs" if l1_ratio == 0 else "saga"
    modelo = LogisticRegression(C=C, l1_ratio=l1_ratio, solver=solver, max_iter=max_iter, tol=tol,
                                random_state=seed)
    return modelo.fit(X, y)


def predecir_pd_logistica(modelo, X):
    """Probabilidad de incumplimiento (clase 1)."""
    return modelo.predict_proba(X)[:, 1]


def _metricas(y, p):
    # provisional mientras no exista metrics.py de P4; la bce es el criterio de selección, igual que en P2
    auc = roc_auc_score(y, p)
    return {"bce": log_loss(y, p, labels=[0, 1]), "auc": auc, "gini": 2 * auc - 1,
            "brier": brier_score_loss(y, p)}


def buscar_regularizacion(X_train, y_train, X_val, y_val, rejilla_C, l1_ratios, sin_penalizacion=True,
                          seed=90, max_iter=1000, ruta_tabla=None, ruta_log=LOG_PATH):
    """Entrena cada (C, l1_ratio), registra el presupuesto y devuelve el de menor bce en validación."""
    configs = [(C, l1) for l1 in l1_ratios for C in rejilla_C]
    if sin_penalizacion: configs.append((np.inf, 0.0))
    filas, mejor, mejor_bce = [], None, np.inf
    for C, l1 in configs:
        inicio = time.perf_counter()
        modelo = entrenar_logistica(X_train, y_train, C, l1, seed, max_iter)
        segundos = time.perf_counter() - inicio
        metricas = _metricas(y_val, predecir_pd_logistica(modelo, X_val))
        # n_params cuenta todos los coeficientes + intercepto; con l1 también importa cuántos quedaron vivos
        metricas["n_no_cero"] = int(np.count_nonzero(modelo.coef_))
        metricas["convergio"] = bool(modelo.n_iter_.max() < max_iter)
        n_params = modelo.coef_.size + 1
        config = {"modelo": "pd_logreg", "C": C, "l1_ratio": l1}
        log_run("pd_logreg", config, seed, n_params, segundos, metricas, log_path=ruta_log)
        filas.append({"C": C, "l1_ratio": l1, **metricas, "n_params": n_params,
                      "segundos": round(segundos, 3)})
        if metricas["bce"] < mejor_bce: mejor, mejor_bce = modelo, metricas["bce"]
    tabla = pd.DataFrame(filas).sort_values("bce").reset_index(drop=True)
    if ruta_tabla is not None: tabla.to_csv(ruta_tabla, index=False)
    return mejor, tabla


def ajustar_platt(p, y):
    """Platt: logística sin penalizar de y sobre logit(p); devuelve el intercepto a y la pendiente b."""
    # la pd llega en float32: sin recortar, una pd de 1.0 exacto daría un logit infinito
    z = logit(np.clip(p, 1e-7, 1 - 1e-7)).reshape(-1, 1)
    modelo = entrenar_logistica(z, y, C=np.inf, l1_ratio=0.0)
    return float(modelo.intercept_[0]), float(modelo.coef_[0, 0])


def aplicar_platt(p, a, b):
    """PD recalibrada: sigmoide(a + b * logit(p))."""
    return expit(a + b * logit(np.clip(p, 1e-7, 1 - 1e-7)))


def recalibrar(modelo, datos, ruta_tabla=None):
    """Ajusta platt en validación y compara la pd original con la recalibrada en validación y prueba."""
    # solo validación entra al ajuste; la prueba mide si la corrección se sostiene en el año siguiente
    a, b = ajustar_platt(predecir_pd_logistica(modelo, datos["X_val"]), datos["y_val"])
    filas = []
    for particion in ["val", "test"]:
        y, p = datos[f"y_{particion}"], predecir_pd_logistica(modelo, datos[f"X_{particion}"])
        for version, q in [("original", p), ("platt", aplicar_platt(p, a, b))]:
            filas.append({"particion": particion, "version": version, "pd_media": float(np.mean(q)),
                          "tasa_real": float(np.mean(y)), **_metricas(y, q), "a": a, "b": b})
    tabla = pd.DataFrame(filas)
    if ruta_tabla is not None: tabla.to_csv(ruta_tabla, index=False)
    return tabla, a, b


def guardar_logistica(modelo, variables, ruta):
    """Guarda el modelo junto con los nombres de las variables (los usarán los puntos de la scorecard)."""
    joblib.dump({"modelo": modelo, "variables": list(variables)}, ruta)


def cargar_logistica(ruta):
    """Devuelve (modelo, variables) sin reentrenar."""
    guardado = joblib.load(ruta)
    return guardado["modelo"], guardado["variables"]


def main():
    parser = build_parser("Regresión logística de referencia para PD", "configs/pd_logreg.yaml")
    parser.add_argument("--recalibrar", action="store_true",
                        help="ajusta platt en validación con el modelo guardado (no reentrena)")
    args = parser.parse_args()
    cfg = init_run(args)
    print("[pd_logreg] config:", cfg)
    datos, busqueda = load_splits("pd", cfg["data"]["scheme"]), cfg["search"]
    if args.recalibrar:
        modelo, _ = cargar_logistica(ROOT / "models" / "pd_logreg.joblib")
        tabla, a, b = recalibrar(modelo, datos, ROOT / "results" / "tables" / "pd_logreg_platt.csv")
        print(tabla.round(4).to_string(index=False))
        print(f"[pd_logreg] platt ajustado en validación: a={a:.4f}, b={b:.4f}")
        return
    modelo, tabla = buscar_regularizacion(
        datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"], busqueda["C"],
        busqueda["l1_ratio"], busqueda["unpenalized"], cfg["seed"], cfg["model"]["max_iter"],
        ROOT / "results" / "tables" / "pd_logreg_busqueda.csv")
    guardar_logistica(modelo, datos["feature_names"], ROOT / "models" / "pd_logreg.joblib")
    print(tabla.to_string(index=False))
    mejor = tabla.iloc[0]
    print(f"[pd_logreg] {len(tabla)} configuraciones en {tabla['segundos'].sum():.1f} s")
    print(f"[pd_logreg] mejor: C={mejor['C']}, l1_ratio={mejor['l1_ratio']}, bce={mejor['bce']:.4f}, "
          f"auc={mejor['auc']:.4f}")


if __name__ == "__main__":
    main()
