"""Regenera TODAS las figuras del informe (31) sin reentrenar ningún modelo. Responsable: P4/P5.

    python -m src.experiments.make_figures                      # todas
    python -m src.experiments.make_figures --lista              # qué figura sale de qué grupo
    python -m src.experiments.make_figures --solo el scorecard  # solo algunos grupos
    python -m src.experiments.make_figures --reentrenar-curva   # permite 1 entrenamiento (~40 s) si falta la curva

Diseño: cada figura sale de los resultados ya guardados (results/tables, models/, data/splits, data/processed).
Los módulos que ya sabían dibujar (evaluate_*, build_scorecard, funciones figura_*) se reutilizan tal cual;
lo que antes se dibujaba dentro de procesos de entrenamiento o de scripts sueltos se redibuja aquí desde las tablas.

Al final se verifica que las 31 figuras existan y que las de los grupos ejecutados se hayan reescrito de verdad.
"""
import importlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT, load_config

TABLAS, FIGURAS = ROOT / "results" / "tables", ROOT / "results" / "figures"
RAW_CSV = ROOT / "data" / "raw" / "loan.csv"
PROCESADOS = ROOT / "data" / "processed"


class Omitir(Exception):
    """El grupo no puede ejecutarse en este entorno (falta un insumo); no es un error del código."""


# ====================================================================== utilidades
def _cfg(nombre: str) -> dict:
    return load_config(str(ROOT / "configs" / nombre))


def _guardar(fig, nombre: str, dpi: int) -> None:
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURAS / nombre, dpi=dpi)
    plt.close(fig)


def _requiere(*rutas: Path) -> None:
    faltan = [str(r.relative_to(ROOT)) for r in rutas if not r.exists()]
    if faltan:
        raise Omitir("faltan insumos: " + ", ".join(faltan))


# ====================================================================== 1. pérdida esperada (5)
def grupo_el(**_):
    """Lee expected_loss.csv y expected_loss_resumen.csv; no carga modelos."""
    _requiere(TABLAS / "expected_loss.csv", TABLAS / "expected_loss_resumen.csv")
    from src.expected_loss.el import expected_loss_report
    from src.experiments.expected_loss import figura_comparacion

    detalle = pd.read_csv(TABLAS / "expected_loss.csv")
    resumen = pd.read_csv(TABLAS / "expected_loss_resumen.csv")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    ead = detalle["ead"].to_numpy()
    # la naive conserva los nombres históricos (distribucion_el.png, el_por_rango_pd.png)
    expected_loss_report(detalle["pd_naive"].to_numpy(), detalle["lgd_naive"].to_numpy(), ead,
                         figures_dir=str(FIGURAS))
    expected_loss_report(detalle["pd_mlp+mlp"].to_numpy(), detalle["lgd_mlp+mlp"].to_numpy(), ead,
                         figures_dir=str(FIGURAS), prefix="mlp_")
    figura_comparacion(resumen, FIGURAS / "el_comparacion_modelos.png")


# ====================================================================== 2. análisis temporal (3)
def grupo_temporal(**_):
    """Tasa de default, LGD media y EAD medio por año de originación (desde data/processed)."""
    _requiere(PROCESADOS / "loans_clean_pd.parquet", PROCESADOS / "loans_clean_lgd.parquet")
    df_pd = pd.read_parquet(PROCESADOS / "loans_clean_pd.parquet")
    df_lgd = pd.read_parquet(PROCESADOS / "loans_clean_lgd.parquet")

    anual = df_pd.groupby("issue_year")["target_pd"].mean().reset_index()
    fig = plt.figure(figsize=(9, 5))
    barras = plt.bar(anual["issue_year"].astype(str), anual["target_pd"] * 100, color="#1f77b4",
                     edgecolor="black", alpha=0.85)
    plt.title("Tasa de Incumplimiento Observada por Año de Originación (PD)", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación (issue_year)", fontsize=11)
    plt.ylabel("Tasa de Default (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for b in barras:
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.5, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.ylim(0, (anual["target_pd"] * 100).max() + 5)
    _guardar(fig, "tasa_default_por_anio.png", 300)

    lgd = df_lgd.groupby("issue_year")["lgd"].mean().reset_index()
    fig = plt.figure(figsize=(9, 5))
    barras = plt.bar(lgd["issue_year"].astype(str), lgd["lgd"] * 100, color="#d62728", edgecolor="black", alpha=0.85)
    plt.title("LGD Media por Año de Originación en Créditos Incumplidos", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación (issue_year)", fontsize=11)
    plt.ylabel("LGD Media (%)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for b in barras:
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height() + 1.0, f"{b.get_height():.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.ylim(0, 105)
    _guardar(fig, "lgd_media_por_anio.png", 300)

    ead = df_pd.groupby("issue_year")["funded_amnt"].agg(["mean", "median"]).reset_index()
    fig = plt.figure(figsize=(9, 5))
    plt.plot(ead["issue_year"].astype(str), ead["mean"], marker="o", linewidth=2.5, color="#9467bd", label="EAD Medio ($)")
    plt.plot(ead["issue_year"].astype(str), ead["median"], marker="s", linestyle="--", linewidth=2,
             color="#8c564b", label="EAD Mediana ($)")
    plt.title("Evolución del Monto Financiado (EAD) por Año de Originación", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación", fontsize=11)
    plt.ylabel("Monto ($ USD)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    _guardar(fig, "ead_medio_por_anio.png", 300)


# ====================================================================== 3. madurez y registros (2)
def _tabla_madurez() -> pd.DataFrame:
    """Resumen por año del dataset CRUDO. Se calcula una vez (necesita loan.csv) y se guarda en results/tables."""
    cache = TABLAS / "madurez_por_anio.csv"
    if cache.exists():
        return pd.read_csv(cache)
    _requiere(RAW_CSV)
    df = pd.read_csv(RAW_CSV, usecols=["issue_d", "loan_status"], low_memory=False)
    df["issue_year"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").dt.year
    resueltos = ["Fully Paid", "Charged Off", "Default",
                 "Does not meet the credit policy. Status:Fully Paid",
                 "Does not meet the credit policy. Status:Charged Off"]
    df["vigente"] = df["loan_status"] == "Current"
    df["resuelto"] = df["loan_status"].isin(resueltos)
    g = df.groupby("issue_year")
    tabla = pd.DataFrame({"total": g.size(), "pct_vigente": g["vigente"].mean() * 100,
                          "pct_resuelto": g["resuelto"].mean() * 100}).reset_index()
    TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(cache, index=False)
    return tabla


def grupo_madurez(**_):
    m = _tabla_madurez()
    x, w = np.arange(len(m)), 0.4
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, m["pct_resuelto"], w, label="Créditos Resueltos (%)", color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.bar(x + w / 2, m["pct_vigente"], w, label="Créditos Vigentes 'Current' (%)", color="#ff7f0e", edgecolor="black", alpha=0.85)
    plt.xticks(x, m["issue_year"].astype(int))
    plt.title("Sesgo de Madurez por Año: Proporción de Créditos Resueltos vs. Vigentes", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación", fontsize=11)
    plt.ylabel("Porcentaje del Total Otorgado (%)", fontsize=11)
    plt.legend(frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    _guardar(fig, "sesgo_madurez.png", 300)


def grupo_registros(**_):
    """REIMPLEMENTADA: ningún script del repo generaba registros_por_anio.png. Compárala con la que cita el blog."""
    m = _tabla_madurez()
    fig = plt.figure(figsize=(9, 5))
    barras = plt.bar(m["issue_year"].astype(int).astype(str), m["total"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.title("Registros por Año de Originación (dataset original)", fontsize=13, fontweight="bold")
    plt.xlabel("Año de Originación (issue_year)", fontsize=11)
    plt.ylabel("Número de créditos", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for b in barras:
        plt.text(b.get_x() + b.get_width() / 2.0, b.get_height(), f"{int(b.get_height()):,}",
                 ha="center", va="bottom", fontsize=8)
    _guardar(fig, "registros_por_anio.png", 300)


# ====================================================================== 4. distribución de la LGD (1)
def grupo_lgd_distribucion(**_):
    """Mismo histograma que src/data/lgd.compute_lgd, pero desde data/processed (no necesita loan.csv)."""
    _requiere(PROCESADOS / "loans_clean_lgd.parquet")
    lgd = pd.read_parquet(PROCESADOS / "loans_clean_lgd.parquet")["lgd"]
    fig = plt.figure(figsize=(9, 5))
    plt.hist(lgd, bins=50, color="#d62728", edgecolor="black", alpha=0.75, density=True)
    plt.title("Distribución Empírica de LGD en Créditos Incumplidos (Opción A)", fontsize=13, fontweight="bold")
    plt.xlabel("Loss Given Default (LGD)", fontsize=11)
    plt.ylabel("Densidad", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.axvline(lgd.mean(), color="black", linestyle="--", linewidth=1.5, label=f"Media: {lgd.mean():.3f}")
    plt.axvline(lgd.median(), color="blue", linestyle=":", linewidth=1.5, label=f"Mediana: {lgd.median():.3f}")
    plt.legend(frameon=True)
    _guardar(fig, "distribucion_lgd.png", 300)


# ====================================================================== 5. fundamento de la pérdida de LGD (3)
def grupo_lgd_teoria(**_):
    """Beta vs normal, QQ y varianza por decil. Reutiliza la función existente (ajusta, no entrena redes)."""
    from src.models.lgd_mlp import figuras_distribucion_lgd
    figuras_distribucion_lgd(FIGURAS)


# ====================================================================== 6. búsqueda y semillas de LGD (3)
def grupo_lgd_busqueda(reentrenar_curva=False, **_):
    from src.data import load_splits

    datos = load_splits("lgd", "random")
    busq = pd.read_csv(TABLAS / "lgd_busqueda.csv")
    orden = busq.groupby("perdida")["rmse"].min().sort_values().index.tolist()   # la ganadora primero

    # (a) distribución predicha vs observada: semilla 0 de cada pérdida + la lineal
    from src.models.lgd_linear import predict_lgd_linear, train_lgd_linear
    lineal, _ = train_lgd_linear(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"])
    p_lineal = predict_lgd_linear(lineal, datos["X_val"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(datos["y_val"], bins=50, density=True, color="lightgray", edgecolor="gray", label="LGD observada")
    for perdida in orden:
        _requiere(TABLAS / f"lgd_pred_val_{perdida}.csv.gz")
        p = pd.read_csv(TABLAS / f"lgd_pred_val_{perdida}.csv.gz")["seed0"]
        ax.hist(p, bins=50, density=True, histtype="step", lw=2, label=f"predicha, mlp {perdida}")
    ax.hist(p_lineal, bins=50, density=True, histtype="step", lw=2, ls="--", label="predicha, lineal")
    ax.set(xlabel="LGD", ylabel="Densidad", title="Distribución de la LGD predicha frente a la observada (validación)")
    ax.legend()
    _guardar(fig, "lgd_pred_vs_obs.png", 300)

    # (b) calibración por deciles: desde la tabla guardada
    _requiere(TABLAS / "lgd_calibracion_deciles.csv")
    d = pd.read_csv(TABLAS / "lgd_calibracion_deciles.csv", header=[0, 1], index_col=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    for k in dict.fromkeys(c[0] for c in d.columns):          # lineal, mse, bce, beta (orden de la tabla)
        ax.plot(d[k]["media_pred"], d[k]["media_obs"], "o-", label=k)
    ax.plot([0.5, 0.9], [0.5, 0.9], "k--", lw=1, label="calibración perfecta")
    ax.set(xlabel="LGD media predicha (decil)", ylabel="LGD media observada (decil)",
           title="Calibración por deciles de la predicción (validación)")
    ax.legend()
    _guardar(fig, "lgd_calibracion_deciles.png", 300)

    # (c) curva de aprendizaje: el historial no se guardaba; se cachea en results/tables
    curva = _historial_curva(datos, busq, reentrenar_curva)
    ganadora = curva["perdida"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curva["epoca"], curva["train_loss"], lw=2, label="entrenamiento (con dropout)")
    ax.plot(curva["epoca"], curva["val_loss"], lw=2, label="validación")
    mejor = int(curva["best_epoch"].iloc[0])
    ax.axvline(mejor, ls="--", color="gray", label=f"mejor época: {mejor}")
    ax.set(xlabel="Época", ylabel=f"Pérdida ({ganadora})", title=f"Curva de aprendizaje del MLP de LGD ({ganadora})")
    ax.legend()
    _guardar(fig, "lgd_curva_aprendizaje.png", 300)


def _historial_curva(datos, busq, reentrenar: bool) -> pd.DataFrame:
    ruta = TABLAS / "lgd_curva_aprendizaje.csv"
    if ruta.exists():
        return pd.read_csv(ruta)
    if not reentrenar:
        raise Omitir("falta results/tables/lgd_curva_aprendizaje.csv (el historial de entrenamiento no se guardó); "
                     "ejecuta una vez con --reentrenar-curva (entrena 1 red, ~40 s) y versiona el CSV")
    from src.models.lgd_mlp import entrenar_lgd_mlp
    from src.search.lgd_random_search import config_de_fila, mejor_por_perdida

    base = _cfg("lgd_mlp.yaml")
    fila = mejor_por_perdida(busq).iloc[0]
    _, _, h = entrenar_lgd_mlp(datos["X_train"], datos["y_train"], datos["X_val"], datos["y_val"],
                               config_de_fila(base, fila), base["run_seeds"][0])
    curva = pd.DataFrame({"epoca": range(1, len(h["train_loss"]) + 1), "train_loss": h["train_loss"],
                          "val_loss": h["val_loss"], "best_epoch": h["best_epoch"], "perdida": fila["perdida"]})
    curva.to_csv(ruta, index=False)
    return curva


# ====================================================================== 7. evaluación de LGD (6)
def grupo_p4_lgd(**_):
    """3 particiones x (distribución + calibración). Lee predicciones guardadas; reutiliza evaluate_lgd.run."""
    from src.experiments import evaluate_lgd
    for yaml_ in ("eval_lgd.yaml", "eval_lgd_test.yaml", "eval_lgd_test_temporal.yaml"):
        evaluate_lgd.run(_cfg(yaml_))


# ====================================================================== 8. calibración de PD (2)
def grupo_p4_pd(**_):
    """Calibración de PD en val y test temporal desde las predicciones guardadas (sin cargar modelos)."""
    from src.experiments import evaluate_pd
    for split in ("val", "test"):
        archivos = {m: f"results/tables/pd_pred_{split}_temporal_{m}.csv.gz" for m in ("logreg", "mlp")}
        _requiere(*[ROOT / a for a in archivos.values()])
        evaluate_pd.run({"eval_pd": {"scheme": "temporal", "split": split, "n_bins": 10,
                                     "modelos": {}, "predicciones": archivos}})


# ====================================================================== 9. scorecard (2)
def grupo_scorecard(**_):
    """Distribución de puntajes y tasa por banda, para la logística y para la red (solo inferencia)."""
    from src.experiments import build_scorecard
    cfg = _cfg("scorecard.yaml")                                    # modelo: mlp
    build_scorecard.run({**cfg, "modelo": "logreg", "ruta_modelo": "models/pd_logreg.joblib"})
    build_scorecard.run(cfg)


# ====================================================================== 10. importancia por permutación (4)
def grupo_importancia(**_):
    """Redibuja desde results/tables/p4_importancia_*.csv (la permutación en sí no se repite)."""
    from src.interpret.permutation import figura_importancia
    csvs = sorted(TABLAS.glob("p4_importancia_*.csv"))
    if not csvs:
        raise Omitir("no hay results/tables/p4_importancia_*.csv; ejecuta interpret_pd e interpret_lgd")
    for csv in csvs:
        cuerpo = csv.stem.removeprefix("p4_importancia_")           # pd_mlp_test_temporal | lgd_lineal_test_random
        tarea, _, resto = cuerpo.partition("_")
        nombre, split, esquema = resto.rsplit("_", 2)
        tabla = pd.read_csv(csv)
        if tarea == "pd":
            figura_importancia(tabla, FIGURAS / f"p4_importancia_{cuerpo}.png",
                               f"Importancia por permutación ({nombre}, {split} {esquema})")
        else:
            figura_importancia(tabla, FIGURAS / f"p4_importancia_{cuerpo}.png",
                               f"Importancia por permutación de LGD ({nombre}, {split} {esquema})",
                               etiqueta="aumento de RMSE al permutar (media ± sd)")


# ====================================================================== catálogo
@dataclass
class Grupo:
    id: str
    descripcion: str
    figuras: list
    fn: Callable
    nota: str = ""


GRUPOS = [
    Grupo("el", "Pérdida esperada (desde expected_loss*.csv)",
          ["distribucion_el.png", "el_por_rango_pd.png", "mlp_distribucion_el.png", "mlp_el_por_rango_pd.png",
           "el_comparacion_modelos.png"], grupo_el),
    Grupo("temporal", "Tasa de default, LGD y EAD por año (data/processed)",
          ["tasa_default_por_anio.png", "lgd_media_por_anio.png", "ead_medio_por_anio.png"], grupo_temporal),
    Grupo("madurez", "Sesgo de madurez (necesita loan.csv una vez; luego usa results/tables/madurez_por_anio.csv)",
          ["sesgo_madurez.png"], grupo_madurez),
    Grupo("registros", "Registros por año (necesita loan.csv una vez)", ["registros_por_anio.png"], grupo_registros,
          "REIMPLEMENTADA: no existía generador; verifica que coincide con lo que cita el blog"),
    Grupo("lgd_dist", "Distribución de la LGD observada (data/processed)", ["distribucion_lgd.png"],
          grupo_lgd_distribucion),
    Grupo("lgd_teoria", "Beta vs normal, QQ y varianza por decil (sección 3.3)",
          ["lgd_ajuste_beta_normal.png", "lgd_qq_beta_normal.png", "lgd_varianza_por_decil.png"], grupo_lgd_teoria),
    Grupo("lgd_busqueda", "Predicha vs observada, calibración por deciles y curva de aprendizaje",
          ["lgd_pred_vs_obs.png", "lgd_calibracion_deciles.png", "lgd_curva_aprendizaje.png"], grupo_lgd_busqueda),
    Grupo("p4_lgd", "Evaluación de LGD (val aleatorio, test aleatorio, test temporal)",
          [f"p4_lgd_{t}_{s}_seed0.png" for s in ("val_random", "test_random", "test_temporal")
           for t in ("distribucion", "calibracion")], grupo_p4_lgd),
    Grupo("p4_pd", "Calibración de PD (val y test temporal)",
          ["p4_pd_calibracion_val_temporal.png", "p4_pd_calibracion_test_temporal.png"], grupo_p4_pd),
    Grupo("scorecard", "Scorecard de la logística y de la red",
          ["p4_scorecard_logreg_test.png", "p4_scorecard_mlp_test.png"], grupo_scorecard,
          "reescribe results/tables/p4_scorecard_* y models/scorecard_referencia_*.npz (mismos valores)"),
    Grupo("importancia", "Importancia por permutación (PD y LGD)",
          ["p4_importancia_pd_logreg_test_temporal.png", "p4_importancia_pd_mlp_test_temporal.png",
           "p4_importancia_lgd_lineal_test_random.png", "p4_importancia_lgd_mlp_bce_seed0_test_random.png"],
          grupo_importancia),
]
FIGURAS_ESPERADAS = [f for g in GRUPOS for f in g.figuras]


# ====================================================================== ejecución y verificación
def run(cfg: dict, solo=None, reentrenar_curva=None) -> list:
    """Ejecuta los grupos, verifica las 31 figuras y devuelve el reporte. Lanza RuntimeError si falta alguna."""
    if reentrenar_curva is None:
        reentrenar_curva = bool(cfg.get("make_figures", {}).get("reentrenar_curva", False))
    ids = [g.id for g in GRUPOS]
    if solo and (desconocidos := [s for s in solo if s not in ids]):
        raise ValueError(f"grupos desconocidos: {desconocidos}. Disponibles: {ids}")

    inicio, reporte = time.time(), []
    for g in GRUPOS:
        if solo and g.id not in solo:
            continue
        t0 = time.perf_counter()
        try:
            g.fn(reentrenar_curva=reentrenar_curva)
            estado, detalle = "OK", ""
        except Omitir as e:
            estado, detalle = "OMITIDO", str(e)
        except Exception as e:                                           # noqa: BLE001
            estado, detalle = "FALLÓ", f"{type(e).__name__}: {e}"
        reporte.append({"grupo": g.id, "estado": estado, "detalle": detalle, "seg": time.perf_counter() - t0})
        print(f"[make_figures] {g.id:<13} {estado:<8} {time.perf_counter() - t0:5.1f} s  {detalle}")

    problemas = []
    ejecutados = {r["grupo"]: r["estado"] for r in reporte}
    for g in GRUPOS:
        if solo and g.id not in solo:
            continue
        for f in g.figuras:
            ruta = FIGURAS / f
            if not ruta.exists():
                problemas.append(f"FALTA {f} (grupo {g.id}, {ejecutados[g.id]})")
            elif ejecutados[g.id] == "OK" and ruta.stat().st_mtime < inicio - 2:
                problemas.append(f"NO SE REESCRIBIÓ {f} (grupo {g.id} dijo OK pero el archivo es anterior)")
    if not solo:
        huerfanas = sorted(p.name for p in FIGURAS.glob("*.png") if p.name not in FIGURAS_ESPERADAS)
        if huerfanas:
            print(f"[make_figures] AVISO: figuras que ningún grupo genera: {huerfanas}")
    fallos = [r for r in reporte if r["estado"] == "FALLÓ"]
    n_ok = sum(1 for f in FIGURAS_ESPERADAS if (FIGURAS / f).exists())
    print(f"[make_figures] {n_ok}/{len(FIGURAS_ESPERADAS)} figuras presentes | "
          f"{sum(r['estado'] == 'OK' for r in reporte)} grupos OK, "
          f"{sum(r['estado'] == 'OMITIDO' for r in reporte)} omitidos, {len(fallos)} fallaron")
    if problemas or fallos:
        raise RuntimeError("make_figures con problemas:\n  " + "\n  ".join(
            problemas + [f"grupo {r['grupo']}: {r['detalle']}" for r in fallos]))
    return reporte


def main():
    p = build_parser("Regenera todas las figuras del informe sin reentrenar", "configs/default.yaml")
    p.add_argument("--solo", nargs="*", metavar="GRUPO", help="ejecuta solo estos grupos (ver --lista)")
    p.add_argument("--lista", action="store_true", help="muestra los grupos y sus figuras, y termina")
    p.add_argument("--reentrenar-curva", action="store_true",
                   help="si falta lgd_curva_aprendizaje.csv, entrena 1 red (~40 s) y lo guarda")
    args = p.parse_args()
    if args.lista:
        for g in GRUPOS:
            print(f"{g.id:<13} {g.descripcion}" + (f"  [{g.nota}]" if g.nota else ""))
            for f in g.figuras:
                print(f"    {f}")
        print(f"\nTotal: {len(FIGURAS_ESPERADAS)} figuras")
        return
    cfg = init_run(args)
    try:
        run(cfg, solo=args.solo, reentrenar_curva=args.reentrenar_curva or None)
    except RuntimeError as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()