"""Búsqueda aleatoria pareada del MLP de LGD: cada configuración se entrena con las tres pérdidas."""
import copy
import itertools
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_splits
from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear
from src.models.lgd_mlp import (cargar_lgd_mlp, contar_parametros, entrenar_lgd_mlp, metricas_lgd,
                                predecir_lgd)
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT
from src.utils.registry import LOG_PATH, config_id, log_run

TABLAS, FIGURAS = ROOT / "results" / "tables", ROOT / "results" / "figures"

ESPACIO = {"hidden": [[32], [64], [64, 32], [128, 64], [128, 64, 32]], "activation": ["relu", "tanh"],
           "dropout": [0.0, 0.1, 0.2, 0.3], "lr": [3e-4, 1e-3, 3e-3], "l2": [0.0, 1e-5, 1e-4, 1e-3]}
PERDIDAS = ["mse", "bce", "beta"]


def muestrear_configuraciones(n, seed):
    """n combinaciones distintas del espacio, elegidas al azar pero reproducibles con la semilla."""
    todas = [dict(zip(ESPACIO, valores)) for valores in itertools.product(*ESPACIO.values())]
    return random.Random(seed).sample(todas, n)


def configurar(base, combinacion, perdida):
    """Copia la config anidada de P5 y le pone los hiperparámetros de la combinación y la pérdida."""
    cfg = copy.deepcopy(base)
    cfg.setdefault("model", {}).update(hidden=combinacion["hidden"], activation=combinacion["activation"],
                                       dropout=combinacion["dropout"])
    cfg.setdefault("train", {}).update(lr=combinacion["lr"], l2=combinacion["l2"])
    cfg["loss"] = {**(cfg.get("loss") if isinstance(cfg.get("loss"), dict) else {}), "name": perdida}
    return cfg


def buscar_lgd(datos, base, n_configs=12, seed=90, ruta_tabla=None, ruta_log=LOG_PATH):
    """Entrena n_configs x 3 pérdidas, registra el presupuesto y devuelve la tabla ordenada por rmse."""
    filas = []
    for i, combinacion in enumerate(muestrear_configuraciones(n_configs, seed)):
        # diseño pareado: la misma combinación con las tres pérdidas, así solo cambia la pérdida
        for perdida in PERDIDAS:
            cfg = configurar(base, combinacion, perdida)
            inicio = time.perf_counter()
            modelo, log_phi, historial = entrenar_lgd_mlp(datos["X_train"], datos["y_train"], datos["X_val"],
                                                          datos["y_val"], cfg, seed)
            segundos = time.perf_counter() - inicio
            p = predecir_lgd(modelo, datos["X_val"])
            metricas = {**metricas_lgd(datos["y_val"], p), "sd_pred": float(p.std())}
            n_params = contar_parametros(modelo, log_phi)
            log_run("lgd_mlp_busqueda", cfg, seed, n_params, segundos, metricas, log_path=ruta_log)
            filas.append({"config": i, "perdida": perdida, "hidden": str(combinacion["hidden"]),
                          "activation": combinacion["activation"], "dropout": combinacion["dropout"],
                          "lr": combinacion["lr"], "l2": combinacion["l2"], **metricas, "n_params": n_params,
                          "epocas": historial["epochs_trained"], "mejor_epoca": historial["best_epoch"],
                          "segundos": round(segundos, 1)})
            print(f"[lgd_busqueda] config {i} {perdida}: rmse={metricas['rmse']:.5f} r2={metricas['r2']:.4f} "
                  f"({segundos:.0f} s)")
    tabla = pd.DataFrame(filas).sort_values("rmse").reset_index(drop=True)
    if ruta_tabla is not None: tabla.to_csv(ruta_tabla, index=False)
    return tabla


def mejor_por_perdida(tabla):
    """La fila de menor rmse de cada pérdida (a estas se les hacen las 30 semillas)."""
    return tabla.loc[tabla.groupby("perdida")["rmse"].idxmin()].sort_values("rmse").reset_index(drop=True)


def config_de_fila(base, fila):
    """Reconstruye la config anidada a partir de una fila de la tabla de búsqueda."""
    combinacion = {"hidden": yaml.safe_load(fila["hidden"]), "activation": fila["activation"],
                   "dropout": float(fila["dropout"]), "lr": float(fila["lr"]), "l2": float(fila["l2"])}
    return configurar(base, combinacion, fila["perdida"])


def escribir_config_final(base, fila, ruta):
    """Escribe la ganadora en el yaml para que run_seeds, la demo y la app usen la configuración final."""
    cfg = config_de_fila(base, fila)
    final = {clave: cfg[clave] for clave in ("data", "train", "model", "loss") if clave in cfg}
    encabezado = ("# configuración final elegida por src/search/lgd_random_search.py\n"
                  f"# menor rmse en validación aleatoria: {fila['rmse']:.5f} (r2 = {fila['r2']:.4f})\n"
                  "# tabla completa en results/tables/lgd_busqueda.csv\n")
    with open(ruta, "w", encoding="utf-8") as archivo:
        archivo.write(encabezado + yaml.safe_dump(final, sort_keys=False, default_flow_style=None))


def semillas_de_config(datos, cfg, semillas, ruta_log=LOG_PATH, experimento=None):
    """Entrena cfg con cada semilla sin guardar modelos; devuelve métricas por semilla y predicciones."""
    perdida, filas, predicciones = cfg["loss"]["name"], [], {}
    experimento = experimento or f"lgd_mlp_semillas_{perdida}"
    for semilla in semillas:
        inicio = time.perf_counter()
        modelo, log_phi, _ = entrenar_lgd_mlp(datos["X_train"], datos["y_train"], datos["X_val"],
                                              datos["y_val"], cfg, semilla)
        segundos = time.perf_counter() - inicio
        p = predecir_lgd(modelo, datos["X_val"])
        metricas = {**metricas_lgd(datos["y_val"], p), "sd_pred": float(p.std())}
        n_params = contar_parametros(modelo, log_phi)
        log_run(experimento, cfg, semilla, n_params, segundos, metricas, log_path=ruta_log)
        filas.append({"semilla": semilla, **metricas, "n_params": n_params, "segundos": round(segundos, 1)})
        predicciones[f"seed{semilla}"] = p
        print(f"[lgd_semillas] {perdida} seed={semilla}: rmse={metricas['rmse']:.5f} ({segundos:.0f} s)")
    return pd.DataFrame(filas), pd.DataFrame(predicciones)


def resumir_semillas(tabla):
    """Media, desviación, mejor y peor de cada métrica entre semillas, como pide la guía."""
    resumen = {"n_semillas": len(tabla), "n_params": int(tabla["n_params"].iloc[0]),
               "minutos": round(tabla["segundos"].sum() / 60, 1)}
    for m in ("rmse", "mae", "r2", "sd_pred"):
        # en rmse y mae lo mejor es lo menor; en r2 lo mayor
        mejor, peor = (tabla[m].max(), tabla[m].min()) if m == "r2" else (tabla[m].min(), tabla[m].max())
        resumen.update({f"{m}_media": tabla[m].mean(), f"{m}_sd": tabla[m].std(), f"{m}_mejor": mejor,
                        f"{m}_peor": peor})
    return resumen


def fase_semillas(base, datos, n_seeds):
    """30 semillas de la mejor config de las pérdidas no ganadoras y sensibilidad a eps de la beta."""
    mejores = mejor_por_perdida(pd.read_csv(TABLAS / "lgd_busqueda.csv"))
    # la ganadora se corre con run_seeds de P5, que guarda los modelos oficiales lgd_mlp_seed{s}.pt
    for _, fila in mejores.iloc[1:].iterrows():
        tabla, predicciones = semillas_de_config(datos, config_de_fila(base, fila), base["run_seeds"][:n_seeds])
        tabla.to_csv(TABLAS / f"lgd_semillas_{fila['perdida']}.csv", index=False)
        # comprimido porque son 30 columnas x 8.823 créditos por pérdida (~2,7 MB sin comprimir)
        predicciones.to_csv(TABLAS / f"lgd_pred_val_{fila['perdida']}.csv.gz", index=False)
    # sensibilidad prerregistrada en la sección 3.3; 1.2e-5 equivale a 0.5 / n_train (smithson y verkuilen)
    cfg_beta, filas = config_de_fila(base, mejores[mejores["perdida"] == "beta"].iloc[0]), []
    for eps in [1e-2, 1e-3, 1e-4, 1.2e-5]:
        cfg_beta["loss"]["eps"] = eps
        modelo, log_phi, _ = entrenar_lgd_mlp(datos["X_train"], datos["y_train"], datos["X_val"],
                                              datos["y_val"], cfg_beta, base["seed"])
        filas.append({"eps": eps, **metricas_lgd(datos["y_val"], predecir_lgd(modelo, datos["X_val"])),
                      "phi": float(np.exp(log_phi.item()))})
    pd.DataFrame(filas).to_csv(TABLAS / "lgd_sensibilidad_eps.csv", index=False)


def fase_resumir(base, datos, n_seeds):
    """Junta las semillas de las tres pérdidas, las compara con la lineal y genera las figuras."""
    mejores = mejor_por_perdida(pd.read_csv(TABLAS / "lgd_busqueda.csv"))
    semillas, ganadora = base["run_seeds"][:n_seeds], mejores.iloc[0]["perdida"]
    # la ganadora viene de run_seeds: se recargan sus modelos y se recuperan los tiempos del registro
    registro = pd.read_csv(LOG_PATH)
    registro = registro[(registro["experiment"] == "lgd_mlp") & (registro["config_id"] == config_id(base))]
    tiempos = registro.drop_duplicates("seed", keep="last").set_index("seed")["seconds"]
    filas, predicciones = [], {}
    for semilla in semillas:
        modelo, log_phi, _ = cargar_lgd_mlp(ROOT / "models" / f"lgd_mlp_seed{semilla}.pt")
        p = predecir_lgd(modelo, datos["X_val"])
        filas.append({"semilla": semilla, **metricas_lgd(datos["y_val"], p), "sd_pred": float(p.std()),
                      "n_params": contar_parametros(modelo, log_phi), "segundos": tiempos.get(semilla, np.nan)})
        predicciones[f"seed{semilla}"] = p
    pd.DataFrame(filas).to_csv(TABLAS / f"lgd_semillas_{ganadora}.csv", index=False)
    pd.DataFrame(predicciones).to_csv(TABLAS / f"lgd_pred_val_{ganadora}.csv.gz", index=False)

    lineal, _ = train_lgd_linear(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"])
    p_lineal = predict_lgd_linear(lineal, datos["X_val"])
    metricas_lineal = {f"{m}_media": v for m, v in metricas_lgd(datos["y_val"], p_lineal).items()}
    resumen = [{"modelo": "lineal (P2, determinista)", **metricas_lineal,
                "sd_pred_media": float(p_lineal.std()), "n_params": datos["X_train"].shape[1] + 1}]
    for perdida in mejores["perdida"]:
        por_semilla = pd.read_csv(TABLAS / f"lgd_semillas_{perdida}.csv")
        resumen.append({"modelo": f"mlp {perdida}", **resumir_semillas(por_semilla)})
    tabla = pd.DataFrame(resumen)
    tabla.to_csv(TABLAS / "lgd_semillas_resumen.csv", index=False)

    # distribución de las predicciones puntuales (semilla 0 de cada pérdida) frente a la observada
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(datos["y_val"], bins=50, density=True, color="lightgray", edgecolor="gray", label="LGD observada")
    for perdida in mejores["perdida"]:
        p = pd.read_csv(TABLAS / f"lgd_pred_val_{perdida}.csv.gz")[f"seed{semillas[0]}"]
        ax.hist(p, bins=50, density=True, histtype="step", lw=2, label=f"predicha, mlp {perdida}")
    ax.hist(p_lineal, bins=50, density=True, histtype="step", lw=2, ls="--", label="predicha, lineal")
    ax.set(xlabel="LGD", ylabel="Densidad",
           title="Distribución de la LGD predicha frente a la observada (validación)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "lgd_pred_vs_obs.png", dpi=300)
    plt.close(fig)

    # curva de entrenamiento y validación de la ganadora (la de P2 trae el título de pd fijo)
    _, _, historial = entrenar_lgd_mlp(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"],
                                       base, semillas[0])
    fig, ax = plt.subplots(figsize=(9, 5))
    epocas = range(1, len(historial["train_loss"]) + 1)
    ax.plot(epocas, historial["train_loss"], lw=2, label="entrenamiento (con dropout)")
    ax.plot(epocas, historial["val_loss"], lw=2, label="validación")
    ax.axvline(historial["best_epoch"], ls="--", color="gray", label=f"mejor época: {historial['best_epoch']}")
    ax.set(xlabel="Época", ylabel=f"Pérdida ({ganadora})",
           title=f"Curva de aprendizaje del MLP de LGD ({ganadora})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "lgd_curva_aprendizaje.png", dpi=300)
    plt.close(fig)
    return tabla


def techo_tanh(modelo):
    """Máxima lgd que puede predecir un mlp cuya última capa oculta es tanh: sigmoide(b + suma |w|)."""
    # cada neurona tanh vale entre -1 y 1, así que el logit no puede pasar de b + suma |w_j|
    salida = modelo.network[-1]
    return torch.sigmoid(salida.bias + salida.weight.abs().sum()).item()


def calibracion_por_decil(y, p):
    """Media predicha y observada en cada decil de la predicción."""
    decil = pd.qcut(p, 10, labels=False) + 1
    tabla = pd.DataFrame({"pred": p, "obs": np.asarray(y, dtype=np.float64), "decil": decil})
    return tabla.groupby("decil").agg(media_pred=("pred", "mean"), media_obs=("obs", "mean"), n=("obs", "size"))


def fase_diagnostico(base, datos, n_seeds=5):
    """Comprueba si el techo de la tanh sesga las medias: techo, saturación, deciles y la misma red con relu."""
    semillas, X_val, y_val = base["run_seeds"], datos["X_val"], datos["y_val"]
    filas = []
    for semilla in semillas:
        modelo, _, _ = cargar_lgd_mlp(ROOT / "models" / f"lgd_mlp_seed{semilla}.pt")
        p = predecir_lgd(modelo, X_val)
        with torch.no_grad(): ocultas = modelo.network[:-1](torch.tensor(X_val)).numpy()
        arriba = p >= np.quantile(p, 0.99)
        filas.append({"semilla": semilla, "techo": techo_tanh(modelo), "max_pred": p.max(),
                      "saturadas_1pct_superior": np.mean(np.abs(ocultas[arriba]) > 0.99),
                      "saturadas_resto": np.mean(np.abs(ocultas[~arriba]) > 0.99)})
    pd.DataFrame(filas).to_csv(TABLAS / "lgd_diagnostico_techo.csv", index=False)

    lineal, _ = train_lgd_linear(datos["X_train"], datos["y_train"], X_val, y_val)
    predicciones = {"lineal": predict_lgd_linear(lineal, X_val)}
    for perdida in PERDIDAS:
        predicciones[perdida] = pd.read_csv(TABLAS / f"lgd_pred_val_{perdida}.csv.gz")["seed0"].to_numpy()
    deciles = pd.concat({k: calibracion_por_decil(y_val, p) for k, p in predicciones.items()}, axis=1)
    deciles.to_csv(TABLAS / "lgd_calibracion_deciles.csv")
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in predicciones: ax.plot(deciles[k]["media_pred"], deciles[k]["media_obs"], "o-", label=k)
    ax.plot([0.5, 0.9], [0.5, 0.9], "k--", lw=1, label="calibración perfecta")
    ax.set(xlabel="LGD media predicha (decil)", ylabel="LGD media observada (decil)",
           title="Calibración por deciles de la predicción (validación)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS / "lgd_calibracion_deciles.png", dpi=300)
    plt.close(fig)

    # misma configuración ganadora pero con relu, que no está acotada: si el techo costara algo, bajaría el rmse
    cfg_relu = copy.deepcopy(base)
    cfg_relu["model"]["activation"] = "relu"
    relu, _ = semillas_de_config(datos, cfg_relu, semillas[:n_seeds], experimento="lgd_mlp_relu")
    tanh = pd.read_csv(TABLAS / f"lgd_semillas_{base['loss']['name']}.csv").head(n_seeds)
    comparacion = pd.DataFrame({"semilla": relu["semilla"], "rmse_tanh": tanh["rmse"].to_numpy(),
                                "rmse_relu": relu["rmse"].to_numpy()})
    comparacion.to_csv(TABLAS / "lgd_techo_relu.csv", index=False)
    return pd.DataFrame(filas), deciles, comparacion


def fase_temporal(base, n_seeds=5):
    """Entrena la config final con la partición temporal (train <= 2013) y la evalúa en validación 2014."""
    # la prueba (2015) no se toca: solo se usan train y validación de la partición temporal
    datos, cfg = load_splits("lgd", "temporal"), copy.deepcopy(base)
    cfg.setdefault("data", {})["scheme"] = "temporal"
    semillas = base["run_seeds"][:n_seeds]
    tabla, predicciones = semillas_de_config(datos, cfg, semillas, experimento="lgd_mlp_temporal")
    lineal, _ = train_lgd_linear(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"])
    p_lineal, y_val = predict_lgd_linear(lineal, datos["X_val"]), datos["y_val"].astype(np.float64)
    filas = [{"modelo": "lineal (P2)", **metricas_lgd(y_val, p_lineal), "media_pred": p_lineal.mean()}]
    for (_, fila), columna in zip(tabla.iterrows(), predicciones.columns):
        filas.append({"modelo": f"mlp {base['loss']['name']} {columna}", "rmse": fila["rmse"],
                      "mae": fila["mae"], "r2": fila["r2"], "media_pred": predicciones[columna].mean()})
    resultado = pd.DataFrame(filas)
    resultado["media_obs_val"], resultado["media_obs_train"] = y_val.mean(), float(datos["y_train"].mean())
    resultado["sesgo"] = resultado["media_pred"] - resultado["media_obs_val"]
    resultado.to_csv(TABLAS / "lgd_estabilidad_temporal.csv", index=False)
    return resultado


def main():
    parser = build_parser("Búsqueda pareada del MLP de LGD", "configs/lgd_mlp.yaml")
    parser.add_argument("--fase", choices=["buscar", "semillas", "resumir", "diagnostico", "temporal"],
                        default="buscar")
    parser.add_argument("--n_configs", type=int, default=12)
    parser.add_argument("--n_seeds", type=int, default=30)
    args = parser.parse_args()
    base = init_run(args)
    print(f"[lgd_busqueda] fase {args.fase}, config base:", base)
    if args.fase == "temporal":
        print(fase_temporal(base).round(4).to_string(index=False))
        return
    datos = load_splits("lgd", base.get("data", {}).get("scheme", "random"))
    if args.fase == "diagnostico":
        for tabla in fase_diagnostico(base, datos): print(tabla.round(4).to_string(), "\n")
        return
    if args.fase == "semillas": return fase_semillas(base, datos, args.n_seeds)
    if args.fase == "resumir":
        print(fase_resumir(base, datos, args.n_seeds).round(5).to_string(index=False))
        return
    tabla = buscar_lgd(datos, base, args.n_configs, base["seed"], TABLAS / "lgd_busqueda.csv")
    mejores = mejor_por_perdida(tabla)
    escribir_config_final(base, mejores.iloc[0], ROOT / "configs" / "lgd_mlp.yaml")
    print(mejores.to_string(index=False))
    print(f"[lgd_busqueda] {len(tabla)} entrenamientos en {tabla['segundos'].sum() / 60:.1f} min; "
          f"ganadora escrita en configs/lgd_mlp.yaml")


if __name__ == "__main__":
    main()
