"""Lógica compartida por la API (app/api.py) y la interfaz Streamlit (app/app.py).

Principio: el formulario NO inventa variables. Se deriva de los artefactos guardados:
  * columnas y categorías válidas  -> models/preprocessor.pkl (el mismo que usaron PD y LGD);
  * valores por defecto            -> mediana (numéricas) y moda (categóricas) del conjunto de entrenamiento;
  * rangos típicos                 -> percentiles 1 y 99 de data/processed/loans_clean_pd.parquet;
  * significado oficial            -> data/raw/LCDataDictionary.xlsx (diccionario del conjunto de Kaggle).

Caso de uso (docs/blog/02_datos_caso_uso.md): admisión / pre-aprobación. Solo se piden datos declarados por el
solicitante y de buró disponibles ANTES del desembolso. Las variables de fuga (grade, int_rate, pagos, recuperaciones…)
no existen en el modelo y por tanto no aparecen aquí.

Unidades: los modelos se entrenaron con LendingClub (USD nominales, 2007-2015), así que el formulario usa USD.
La conversión a COP es solo informativa (la hace la interfaz con la tasa de /config).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.lgd_mlp import cargar_lgd_mlp, predecir_lgd  # noqa: E402
from src.scorecard.api import ScorecardAPI  # noqa: E402

PROCESADOS = ROOT / "data" / "processed" / "loans_clean_pd.parquet"
DICCIONARIO = ROOT / "data" / "raw" / "LCDataDictionary.xlsx"

# --------------------------------------------------------------------------- metadatos de presentación
# (etiqueta, grupo, nivel, unidad, ayuda). Solo texto de interfaz; categorías, rangos y defectos salen de los artefactos.
META: dict[str, tuple[str, str, str, str, str]] = {
    "loan_amnt": ("Monto solicitado", "Crédito solicitado", "basico", "USD",
                  "Monto del crédito. Se asume financiado por completo: funded_amnt = funded_amnt_inv = monto (esa es también la EAD)."),
    "term": ("Plazo", "Crédito solicitado", "basico", "", "Número de pagos del crédito: 36 o 60 meses."),
    "purpose": ("Propósito del crédito", "Crédito solicitado", "basico", "", "Categoría declarada por el solicitante para el préstamo."),
    "annual_inc": ("Ingreso anual declarado", "Perfil del solicitante", "basico", "USD", "Ingreso anual autorreportado al registrarse."),
    "emp_length": ("Antigüedad laboral", "Perfil del solicitante", "basico", "",
                   "Años de empleo (0 = menos de un año, 10 = diez o más). «No informa» se trata como dato faltante."),
    "home_ownership": ("Vivienda", "Perfil del solicitante", "basico", "", "Situación de vivienda declarada en el registro."),
    "verification_status": ("Verificación del ingreso", "Perfil del solicitante", "avanzado", "",
                            "Si el ingreso fue verificado por el prestamista. (Esta variable no figura en LCDataDictionary.xlsx; "
                            "se documenta por sus valores en los datos.)"),
    "addr_state": ("Estado (EE. UU.)", "Perfil del solicitante", "avanzado", "",
                   "Estado de residencia declarado. Los estados con menos del 1 % de los datos se agrupan en una sola categoría."),
    "dti": ("Relación deuda/ingreso (DTI)", "Capacidad de pago", "basico", "%",
            "Pagos mensuales de deuda (sin hipoteca ni este crédito) sobre el ingreso mensual declarado."),
    "revol_util": ("Uso del crédito rotativo", "Capacidad de pago", "basico", "%",
                   "Porcentaje del cupo rotativo disponible que está en uso (puede superar 100 %)."),
    "revol_bal": ("Saldo rotativo total", "Capacidad de pago", "avanzado", "USD", "Saldo total de las líneas rotativas (tarjetas)."),
    "total_rev_hi_lim": ("Cupo rotativo total", "Capacidad de pago", "avanzado", "USD", "Cupo/límite máximo total de las líneas rotativas."),
    "tot_cur_bal": ("Saldo actual de todas las cuentas", "Capacidad de pago", "avanzado", "USD", "Saldo corriente total de todas las cuentas."),
    "total_acc": ("Líneas de crédito totales", "Historial de buró", "avanzado", "n", "Total de líneas de crédito que aparecen en el buró."),
    "open_acc": ("Líneas de crédito abiertas", "Historial de buró", "avanzado", "n", "Número de líneas de crédito abiertas."),
    "inq_last_6mths": ("Consultas al buró (6 meses)", "Historial de buró", "avanzado", "n",
                       "Consultas en los últimos 6 meses (sin contar crédito de auto ni hipotecario)."),
    "delinq_2yrs": ("Moras de 30+ días (2 años)", "Historial de buró", "avanzado", "n", "Eventos de mora de 30 o más días en los últimos 2 años."),
    "pub_rec": ("Registros públicos negativos", "Historial de buró", "avanzado", "n", "Número de registros públicos despectivos."),
    "collections_12_mths_ex_med": ("Cobranzas (12 meses, sin médicas)", "Historial de buró", "avanzado", "n",
                                   "Cobranzas en los últimos 12 meses excluyendo las médicas."),
    "acc_now_delinq": ("Cuentas actualmente en mora", "Historial de buró", "avanzado", "n", "Cuentas en las que el solicitante está hoy en mora."),
    "tot_coll_amt": ("Monto total en cobranza", "Historial de buró", "avanzado", "USD", "Monto total en cobranza alguna vez adeudado."),
}
ORDEN_GRUPOS = ["Crédito solicitado", "Perfil del solicitante", "Capacidad de pago", "Historial de buró"]

# Variables que el modelo exige pero el usuario no controla; se fijan y se documentan.
DERIVADAS = {"funded_amnt": "loan_amnt", "funded_amnt_inv": "loan_amnt"}
FIJAS = {"policy_code": 1.0, "pymnt_plan": "n", "application_type": "INDIVIDUAL"}
# earliest_cr_line: en preprocessor.pkl TODAS sus 601 categorías (mes-año) quedaron como «infrecuentes» (<1 %) y se
# agrupan en una sola columna; cambiarla no altera el modelo. Por eso no se pregunta (ver docs, hallazgo de datos).
SIN_EFECTO = ["earliest_cr_line"]

ETIQUETAS = {
    "purpose": {"debt_consolidation": "Consolidar deudas", "credit_card": "Pagar tarjeta de crédito", "home_improvement": "Mejoras del hogar",
                "other": "Otro", "major_purchase": "Compra grande", "small_business": "Negocio pequeño", "car": "Vehículo",
                "medical": "Gastos médicos", "moving": "Mudanza", "vacation": "Vacaciones", "wedding": "Boda", "house": "Casa",
                "educational": "Educación", "renewable_energy": "Energía renovable"},
    "home_ownership": {"RENT": "Arriendo", "MORTGAGE": "Hipoteca", "OWN": "Propia", "OTHER": "Otra", "NONE": "Ninguna"},
    "verification_status": {"Not Verified": "No verificado", "Source Verified": "Fuente verificada", "Verified": "Verificado"},
    "term": {" 36 months": "36 meses", " 60 months": "60 meses"},
    "emp_length": {"< 1 year": "Menos de 1 año", "1 year": "1 año", "10+ years": "10 años o más", "missing": "No informa"},
}
# Enteros (conteos) frente a continuas
ENTEROS = {"delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec", "total_acc", "collections_12_mths_ex_med", "acc_now_delinq"}
# Cotas duras (valores imposibles → error 422). Fuera de p1–p99 pero dentro de la cota solo genera aviso.
COTAS = {"loan_amnt": (500, 35_000), "annual_inc": (1, 10_000_000), "dti": (0, 100), "revol_util": (0, 500)}


def _f(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}".rstrip("0").rstrip(".")


# --------------------------------------------------------------------------- recursos
class Servicio:
    """Carga una sola vez scorecard (PD), modelo LGD y esquema del formulario."""

    def __init__(self):
        seed = int(os.getenv("ME_PRESTAS_LGD_SEED", "90"))
        preferido = os.getenv("ME_PRESTAS_LGD_SCHEME", "temporal").strip().lower()
        if preferido not in ("temporal", "random"):
            raise ValueError("ME_PRESTAS_LGD_SCHEME debe ser 'temporal' o 'random'.")
        self.sc = ScorecardAPI.cargar()
        self.aviso_lgd = None
        for esquema in (preferido, "random" if preferido == "temporal" else "temporal"):
            ruta = ROOT / "models" / f"lgd_mlp{'' if esquema == 'random' else '_' + esquema}_seed{seed}.pt"
            if ruta.exists():
                break
        else:
            raise FileNotFoundError(f"No hay modelo LGD para la semilla {seed} en models/.")
        if esquema != preferido:
            self.aviso_lgd = f"Se usa el LGD '{esquema}' porque no existe el del esquema '{preferido}' (la PD es temporal)."
        self.lgd, _, self.perdida_lgd = cargar_lgd_mlp(ruta)
        self.esquema_lgd, self.ruta_lgd, self.semilla_lgd = esquema, str(ruta.relative_to(ROOT)), seed
        self.campos = self._construir_esquema()
        self.por_clave = {c["clave"]: c for c in self.campos}
        self.defectos = self._defectos_completos()
        self._hist = None

    # ---- esquema derivado de los artefactos
    def _estadisticos(self):
        """Mediana/moda y percentiles 1-99 desde los datos procesados de entrenamiento (solo originaciones del train temporal)."""
        import pandas as pd
        out = {"mediana": {}, "moda": {}, "p01": {}, "p99": {}}
        try:
            df = pd.read_parquet(PROCESADOS)
            df = df[df["issue_year"] <= 2013] if "issue_year" in df.columns else df
        except Exception:
            return out
        for nombre, _, cols in self.sc.preprocesador.transformers_:
            for c in cols if nombre in ("num", "cat") else []:
                if c not in df.columns:
                    continue
                if nombre == "num":
                    out["mediana"][c] = float(df[c].median())
                    out["p01"][c], out["p99"][c] = float(df[c].quantile(.01)), float(df[c].quantile(.99))
                else:
                    m = df[c].mode(dropna=True)
                    if len(m):
                        out["moda"][c] = str(m.iloc[0])
        return out

    def _descripciones_oficiales(self):
        try:
            import pandas as pd
            d = pd.read_excel(DICCIONARIO)
            return {str(k).strip(): str(v).strip() for k, v in zip(d.iloc[:, 0], d.iloc[:, 1])}
        except Exception:
            return {}

    def _construir_esquema(self):
        sc = self.sc
        est, dic = self._estadisticos(), self._descripciones_oficiales()
        categorias = sc.categorias()
        medianas_imp = {}
        for nombre, trans, cols in sc.preprocesador.transformers_:
            if nombre == "num":
                imp = trans.named_steps["imputer"]
                medianas_imp = {c: float(v) for c, v in zip(cols, imp.statistics_)}
        campos = []
        for clave, (etiqueta, grupo, nivel, unidad, ayuda) in META.items():
            if clave not in sc.variables_requeridas:
                continue  # el modelo guardado no usa esta variable: no se pregunta
            esnum = clave in medianas_imp
            campo = {"clave": clave, "etiqueta": etiqueta, "grupo": grupo, "nivel": nivel, "unidad": unidad, "ayuda": ayuda,
                     "diccionario": dic.get(clave), "tipo": "numero" if esnum else "categoria"}
            if esnum:
                lo, hi = est["p01"].get(clave), est["p99"].get(clave)
                cota = COTAS.get(clave, (0, None))
                campo.update(defecto=est["mediana"].get(clave, medianas_imp[clave]), p01=lo, p99=hi, entero=clave in ENTEROS,
                             min=cota[0], max=cota[1])
            else:
                opciones = categorias.get(clave, [])
                etiq = ETIQUETAS.get(clave, {})
                campo["opciones"] = [{"valor": o, "etiqueta": etiq.get(o, o.strip())} for o in opciones]
                moda = est["moda"].get(clave)
                campo["defecto"] = moda if moda in opciones else (opciones[0] if opciones else "")
            campos.append(campo)
        campos.sort(key=lambda c: (ORDEN_GRUPOS.index(c["grupo"]), list(META).index(c["clave"])))
        return campos

    def _defectos_completos(self):
        """Valores para TODAS las variables que exige el preprocesador (las no preguntadas quedan fijas y documentadas)."""
        d = {c["clave"]: c["defecto"] for c in self.campos}
        d.update(FIJAS)
        est_cat = {}
        for nombre, _, cols in self.sc.preprocesador.transformers_:
            if nombre == "cat":
                est_cat = {c: None for c in cols}
        d.setdefault("initial_list_status", self._estadisticos()["moda"].get("initial_list_status", "f"))
        for c in SIN_EFECTO:
            d.setdefault(c, "missing")
        for c in est_cat:
            d.setdefault(c, "missing")
        return d

    # ---- validación y armado de la fila de entrada
    def completar(self, datos: dict[str, Any]):
        """datos (subconjunto de claves del formulario) -> (fila completa para el modelo, avisos, claves tomadas por defecto)."""
        desconocidas = sorted(set(datos) - set(self.por_clave))
        if desconocidas:
            raise ValueError(f"Variables no reconocidas: {desconocidas}. Válidas: {sorted(self.por_clave)}")
        fila, avisos, por_defecto = dict(self.defectos), [], []
        for clave, campo in self.por_clave.items():
            valor = datos.get(clave)
            if valor is None or valor == "":
                por_defecto.append(clave)
                continue
            if campo["tipo"] == "numero":
                try:
                    v = float(valor)
                except (TypeError, ValueError):
                    raise ValueError(f"{campo['etiqueta']}: se esperaba un número.") from None
                if not np.isfinite(v):
                    raise ValueError(f"{campo['etiqueta']}: valor no finito.")
                lo, hi = campo["min"], campo["max"]
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    raise ValueError(f"{campo['etiqueta']}: {v:g} fuera del rango admitido [{lo}, {hi if hi is not None else '∞'}].")
                if campo["entero"] and v != int(v):
                    raise ValueError(f"{campo['etiqueta']}: debe ser un entero.")
                p01, p99 = campo.get("p01"), campo.get("p99")
                if p01 is not None and not (p01 <= v <= p99):
                    avisos.append(f"{campo['etiqueta']} = {_f(v)} está fuera del rango típico de entrenamiento "
                                  f"({_f(p01)} a {_f(p99)}, percentiles 1 a 99): la estimación es una extrapolación.")
                fila[clave] = v
            else:
                valores = [o["valor"] for o in campo["opciones"]]
                if str(valor) not in valores:
                    raise ValueError(f"{campo['etiqueta']}: «{valor}» no es una categoría conocida por el modelo ({valores}).")
                fila[clave] = str(valor)
        # derivadas y coherencias entre campos
        for destino, origen in DERIVADAS.items():
            fila[destino] = fila[origen]
        if fila["open_acc"] > fila["total_acc"]:
            avisos.append("Las líneas abiertas superan las líneas totales: revisa ambos datos.")
        limite, saldo = fila["total_rev_hi_lim"], fila["revol_bal"]
        if "revol_util" not in por_defecto and "total_rev_hi_lim" not in por_defecto and "revol_bal" not in por_defecto and limite > 0:
            implicito = 100 * saldo / limite
            if abs(implicito - fila["revol_util"]) > 15:
                avisos.append(f"El uso rotativo ({fila['revol_util']:.0f} %) no coincide con saldo/cupo ({implicito:.0f} %); "
                              "en los datos de entrenamiento ambos son consistentes.")
        return fila, avisos, por_defecto

    # ---- cálculo
    def calcular(self, datos: dict[str, Any]) -> dict[str, Any]:
        fila, avisos, por_defecto = self.completar(datos)
        X = self.sc.preprocesar(fila)
        puntaje, pd_ = self.sc.score(X)
        puntaje, pd_ = float(puntaje[0]), float(pd_[0])
        lgd = float(predecir_lgd(self.lgd, X)[0])
        ead = float(fila["funded_amnt"])  # EAD = funded_amnt al originar (src/data/lgd.py)
        percentil = float(self.sc.percentile([puntaje])[0])
        if self.aviso_lgd:
            avisos.append(self.aviso_lgd)
        return {"PD": pd_, "Puntaje": puntaje, "Percentil": percentil, "LGD": lgd, "EAD_USD": ead,
                "Perdida_esperada_USD": pd_ * lgd * ead,
                "banda_riesgo": "alto" if percentil < 33.3 else "medio" if percentil < 66.7 else "bajo",
                "por_defecto": por_defecto, "avisos": avisos, "split_lgd": self.esquema_lgd}

    def escenario(self, datos: dict[str, Any], cambios: dict[str, Any]) -> dict[str, Any]:
        if not cambios:
            raise ValueError("Indica al menos una característica a cambiar.")
        base = self.calcular(datos)
        nuevo = self.calcular({**datos, **cambios})
        claves = ("Puntaje", "PD", "Percentil", "LGD", "Perdida_esperada_USD")
        return {"base": base, "escenario": nuevo, "cambio": {k: nuevo[k] - base[k] for k in claves}}

    def histograma(self, n_bins: int = 40):
        """Distribución de puntajes de la población de referencia (entrenamiento) para ubicar al usuario."""
        if self._hist is None:
            ref = self.sc._referencia
            lo, hi = np.quantile(ref, [0.005, 0.995])
            cuentas, bordes = np.histogram(np.clip(ref, lo, hi), bins=n_bins, range=(lo, hi))
            self._hist = {"bordes": bordes.tolist(), "cuentas": cuentas.tolist(), "n": int(ref.size),
                          "p25": float(np.quantile(ref, .25)), "mediana": float(np.quantile(ref, .5)), "p75": float(np.quantile(ref, .75))}
        return self._hist


_servicio: Servicio | None = None
_lock = threading.Lock()


def obtener_servicio() -> Servicio:
    global _servicio
    with _lock:
        if _servicio is None:
            _servicio = Servicio()
        return _servicio
