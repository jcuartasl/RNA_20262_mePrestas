"""API HTTP de ¿Me prestas?: conecta index.html con los modelos guardados del proyecto.

Un solo comando, desde la raíz del repositorio:
    python -m app.api                      # abre http://127.0.0.1:8000
    (equivalente: uvicorn app.api:app --host 127.0.0.1 --port 8000)

La propia API sirve index.html en "/" y las figuras en "/figuras", así que no hace
falta Live Server ni configurar CORS. No reentrena: solo carga modelos de models/.

El formulario se deriva de los modelos guardados (GET /esquema): mismas 21 variables que el preprocesador exige
al usuario, en USD (unidad de entrenamiento). Las variables no preguntadas se fijan y documentan en servicio.py.

Endpoints: GET /esquema, /poblacion, /config, /metricas, /salud · POST /predecir {datos}, /escenario {datos, cambios}

Variables de entorno (todas opcionales):
    ME_PRESTAS_LGD_SEED      semilla del modelo LGD guardado (por defecto 90)
    ME_PRESTAS_LGD_SCHEME    temporal | random (por defecto temporal, coherente con PD y scorecard)
    ME_PRESTAS_COP_PER_USD   tasa COP/USD solo informativa para mostrar equivalentes (por defecto 4000)
    ME_PRESTAS_REPO_URL / ME_PRESTAS_REPORT_URL / ME_PRESTAS_VIDEO_URL   enlaces que muestra la página
    ME_PRESTAS_ALLOWED_ORIGINS   orígenes extra permitidos por CORS, separados por coma
    ME_PRESTAS_HOST / ME_PRESTAS_PORT   host y puerto de `python -m app.api`
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.servicio import obtener_servicio  # noqa: E402

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
INDEX_HTML = ROOT / "index.html"

app = FastAPI(title="¿Me prestas? API", version="3.0.0")
_origins = [x.strip() for x in os.getenv(
    "ME_PRESTAS_ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500"
).split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

Valor = Union[float, str, None]


class Solicitud(BaseModel):
    """Claves = variables del formulario (GET /esquema). Las omitidas toman mediana/moda del entrenamiento."""
    datos: dict[str, Valor] = Field(default_factory=dict)


class Escenario(Solicitud):
    cambios: dict[str, Valor] = Field(min_length=1)


def _fx() -> float:
    """Tasa COP/USD SOLO informativa: los modelos trabajan en USD (LendingClub 2007-2015)."""
    fx = float(os.getenv("ME_PRESTAS_COP_PER_USD", "4000"))
    if not np.isfinite(fx) or fx <= 0:
        raise ValueError("ME_PRESTAS_COP_PER_USD debe ser una tasa numérica positiva.")
    return fx


def _servicio():
    try:
        return obtener_servicio()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"No se pudieron cargar los modelos guardados: {exc}") from exc


# ----------------------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def inicio():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="No se encontró index.html en la raíz del repositorio.")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/salud")
def health():
    r = _servicio()
    return {"estado": "ok", "scorecard": "cargado", "lgd": r.perdida_lgd, "split": r.esquema_lgd,
            "lgd_modelo": r.ruta_lgd, "aviso": r.aviso_lgd, "variables_formulario": len(r.campos)}


@app.get("/esquema")
def esquema():
    """Campos del formulario derivados de preprocessor.pkl: tipos, categorías válidas, defectos y rangos de entrenamiento."""
    r = _servicio()
    return {"campos": r.campos, "grupos": sorted({c["grupo"] for c in r.campos}),
            "fijas": {"funded_amnt / funded_amnt_inv": "igual al monto solicitado (EAD)", "policy_code": 1,
                      "pymnt_plan": "n", "application_type": "INDIVIDUAL"},
            "sin_efecto": ["earliest_cr_line"]}


@app.get("/poblacion")
def poblacion():
    """Histograma de puntajes de la población de referencia (entrenamiento) para ubicar al usuario."""
    return _servicio().histograma()


@app.get("/config")
def config():
    """Enlaces finales y parámetros de la scorecard que muestra la página."""
    out: dict[str, Any] = {
        "repo_url": os.getenv("ME_PRESTAS_REPO_URL", "https://github.com/jcuartasl/RNA_20262_mePrestas").strip(),
        "report_url": os.getenv("ME_PRESTAS_REPORT_URL", "").strip(),
        "video_url": os.getenv("ME_PRESTAS_VIDEO_URL", "https://youtu.be/Li35oMpmqmk").strip(),
        "tasa_COP_USD": _fx(),
        "moneda_modelo": "USD",
    }
    try:
        sc = _servicio().sc.scorecard
        out["scorecard"] = {"puntaje_base": sc.puntaje_base, "odds_base": sc.odds_base, "pdo": sc.pdo}
    except Exception:
        pass
    return out


def _leer_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@app.get("/metricas")
def metricas():
    """Métricas reales de PD en la prueba temporal, leídas de results/tables (nunca fijas en el HTML)."""
    try:
        resumen = _leer_csv(TABLES / "p4_pd_eval_test_temporal_resumen.csv")
        justa = _leer_csv(TABLES / "p4_comparacion_justa_pd.csv")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Falta una tabla de resultados: {exc.filename}") from exc
    extra = {}
    for fila in justa:
        clave = "logreg" if "log" in fila["modelo"].lower() else "mlp"
        extra[clave] = {"parametros": _f(fila.get("parametros")), "configs": _f(fila.get("configs_busqueda")),
                        "minutos": _f(fila.get("minutos")), "n_semillas_val": _f(fila.get("n_semillas_val")),
                        "n_semillas_test": _f(fila.get("n_semillas_test")),
                        "auc_val_sd": _f(fila.get("auc_val_sd"))}
    salida = {}
    for fila in resumen:
        k = fila["modelo"]
        salida[k] = {m: _f(fila.get(f"{m}_media")) for m in ("auc", "gini", "ks", "brier")} | extra.get(k, {})
    return {"particion": "prueba temporal (originaciones de 2015)",
            "fuente": ["results/tables/p4_pd_eval_test_temporal_resumen.csv",
                       "results/tables/p4_comparacion_justa_pd.csv"],
            "modelos": salida}


@app.post("/predecir")
def predict(req: Solicitud):
    r = _servicio()
    try:
        return r.calcular(req.datos)
    except ValueError as exc:  # entrada fuera de esquema: error del cliente
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo calcular la predicción: {exc}") from exc


@app.post("/escenario")
def escenario(req: Escenario):
    """Compara el caso base con el mismo caso cambiando una o más características."""
    r = _servicio()
    try:
        return r.escenario(req.datos, req.cambios)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"No se pudo calcular el escenario: {exc}") from exc


# Figuras generadas por make_figures.py / evaluate_*.py (solo lectura).
app.mount("/figuras", StaticFiles(directory=str(FIGURES), check_dir=False), name="figuras")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=os.getenv("ME_PRESTAS_HOST", "127.0.0.1"),
                port=int(os.getenv("ME_PRESTAS_PORT", "8000")))