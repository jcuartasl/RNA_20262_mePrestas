"""Interfaz Streamlit de ¿Me prestas? (alternativa a index.html + API).

Ejecutar desde la raíz del repositorio:   streamlit run app/app.py
Usa EXACTAMENTE el mismo servicio que la API (app/servicio.py): mismos campos, validaciones y modelos guardados.
No reentrena. Variables opcionales: ME_PRESTAS_LGD_SEED (90), ME_PRESTAS_LGD_SCHEME (temporal),
ME_PRESTAS_COP_PER_USD (4000, solo informativa), ME_PRESTAS_REPORT_URL, ME_PRESTAS_VIDEO_URL, ME_PRESTAS_REPO_URL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT_LOCAL = Path(__file__).resolve().parents[1]
if str(ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(ROOT_LOCAL))

from app.servicio import obtener_servicio  # noqa: E402

st.set_page_config(page_title="¿Me prestas? · Riesgo de crédito", page_icon="📊", layout="wide")
FX = float(os.environ.get("ME_PRESTAS_COP_PER_USD", "3140"))


@st.cache_resource(show_spinner="Cargando modelos guardados…")
def recursos():
    return obtener_servicio()


def usd(x):
    return f"US${x:,.0f}"


def widget(campo, clave_ui, opcional):
    """Dibuja el control de un campo del esquema y devuelve su valor (None si se deja vacío)."""
    etiqueta = campo["etiqueta"] + (f" ({campo['unidad']})" if campo["unidad"] not in ("", "n") else "")
    ayuda = campo["ayuda"] + (f"\n\nDiccionario oficial: {campo['diccionario']}" if campo.get("diccionario") else "")
    if campo["tipo"] == "categoria":
        valores = [o["valor"] for o in campo["opciones"]]
        nombres = {o["valor"]: o["etiqueta"] for o in campo["opciones"]}
        if opcional:
            valores = [None] + valores
            nombres[None] = "— valor típico —"
            idx = 0
        else:
            idx = valores.index(campo["defecto"]) if campo["defecto"] in valores else 0
        return st.selectbox(etiqueta, valores, index=idx, format_func=lambda v: nombres[v], help=ayuda, key=clave_ui)
    paso = 1.0 if campo["entero"] else 0.1
    lo = campo["min"] if campo["min"] is not None else 0.0
    if opcional:
        txt = st.text_input(etiqueta, value="", placeholder=f"típico: {campo['defecto']:,.4g}", help=ayuda, key=clave_ui)
        if txt.strip() == "":
            return None
        try:
            return float(txt.replace(",", "."))
        except ValueError:
            st.error(f"{campo['etiqueta']}: escribe un número.")
            return None
    return st.number_input(etiqueta, min_value=float(lo), max_value=float(campo["max"]) if campo["max"] is not None else None,
                           value=float(round(campo["defecto"], 2)), step=1000.0 if campo["unidad"] == "USD" else paso, help=ayuda, key=clave_ui)


def mostrar(r, titulo):
    st.subheader(titulo)
    cols = st.columns(5)
    for col, (e, v) in zip(cols, (("PD", f"{r['PD']:.2%}"), ("Puntaje", f"{r['Puntaje']:.1f}"), ("Percentil", f"{r['Percentil']:.1f}"),
                                  ("LGD", f"{r['LGD']:.1%}"), ("Pérdida esperada", usd(r["Perdida_esperada_USD"])))):
        col.metric(e, v)
    st.caption(f"E[L] = PD × LGD × EAD = {r['PD']:.4f} × {r['LGD']:.4f} × {usd(r['EAD_USD'])} "
               f"(≈ ${r['Perdida_esperada_USD'] * FX:,.0f} COP a {FX:,.0f} COP/USD, solo informativo). Riesgo {r['banda_riesgo']} (tercil de puntaje en el entrenamiento).")
    for a in r["avisos"]:
        st.warning(a)


def main():
    st.title("¿Me prestas?")
    st.write("Estimación de riesgo de crédito con los modelos guardados del proyecto (pre-aprobación: datos disponibles antes del desembolso).")
    st.info("Prototipo académico con datos de Lending Club (EE. UU., 2007–2015) en dólares; no es una decisión real de crédito.")
    try:
        s = recursos()
    except Exception as exc:
        st.error(f"No se pudieron cargar los artefactos del proyecto: {exc}")
        st.stop()
    sc = s.sc.scorecard
    st.caption(f"Scorecard: base {sc.puntaje_base:g}, odds {sc.odds_base:g}:1, PDO {sc.pdo:g}. LGD: {s.perdida_lgd}, split {s.esquema_lgd}, `{s.ruta_lgd}`.")
    if s.aviso_lgd:
        st.warning(s.aviso_lgd)

    with st.form("solicitud"):
        st.subheader("Datos del solicitante y del crédito (USD)")
        datos: dict = {}
        grupos = sorted({c["grupo"] for c in s.campos}, key=lambda g: [c["grupo"] for c in s.campos].index(g))
        for g in grupos:
            basicos = [c for c in s.campos if c["grupo"] == g and c["nivel"] == "basico"]
            if basicos:
                st.markdown(f"**{g}**")
                cols = st.columns(2)
                for i, c in enumerate(basicos):
                    with cols[i % 2]:
                        datos[c["clave"]] = widget(c, f"f_{c['clave']}", opcional=False)
        with st.expander("Datos avanzados de buró y perfil (opcionales)"):
            st.caption("Si los dejas vacíos se usa la mediana o moda del entrenamiento; el resultado indica cuáles fueron.")
            cols = st.columns(2)
            for i, c in enumerate([c for c in s.campos if c["nivel"] == "avanzado"]):
                with cols[i % 2]:
                    datos[c["clave"]] = widget(c, f"f_{c['clave']}", opcional=True)
        enviado = st.form_submit_button("Calcular estimación", type="primary")

    if enviado:
        limpio = {k: v for k, v in datos.items() if v is not None}
        try:
            st.session_state["datos"] = limpio
            st.session_state["base"] = s.calcular(limpio)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"No se pudo calcular el caso. Detalle: {exc}")

    base, datos_base = st.session_state.get("base"), st.session_state.get("datos")
    if base:
        mostrar(base, "Estimación del caso base")
        if base["por_defecto"]:
            with st.expander(f"{len(base['por_defecto'])} dato(s) completados con el valor típico del entrenamiento"):
                st.write(", ".join(s.por_clave[k]["etiqueta"] for k in base["por_defecto"]))
        hist = s.histograma()
        import pandas as pd
        centros = [(a + b) / 2 for a, b in zip(hist["bordes"][:-1], hist["bordes"][1:])]
        st.caption(f"Distribución de puntajes de la población de entrenamiento (n = {hist['n']:,}); tu puntaje: {base['Puntaje']:.1f}.")
        st.bar_chart(pd.DataFrame({"personas": hist["cuentas"]}, index=[round(c) for c in centros]))

        st.divider()
        st.subheader("¿Qué pasaría si cambias una característica?")
        claves = [c["clave"] for c in s.campos]
        clave = st.selectbox("Característica", claves, format_func=lambda k: s.por_clave[k]["etiqueta"], key="wi_var")
        campo = s.por_clave[clave]
        actual = datos_base.get(clave, campo["defecto"])
        if campo["tipo"] == "categoria":
            valores = [o["valor"] for o in campo["opciones"]]
            nombres = {o["valor"]: o["etiqueta"] for o in campo["opciones"]}
            nuevo = st.selectbox("Nuevo valor", valores, index=valores.index(actual) if actual in valores else 0,
                                 format_func=lambda v: nombres[v], key=f"wi_{clave}")
        else:
            nuevo = st.number_input("Nuevo valor", value=float(actual), step=1.0 if campo["entero"] else 0.1, key=f"wi_{clave}")
        if st.button("Comparar escenario"):
            try:
                e = s.escenario(datos_base, {clave: nuevo})
                izq, der = st.columns(2)
                with izq:
                    mostrar(e["base"], "Caso base")
                with der:
                    mostrar(e["escenario"], "Escenario modificado")
                c = e["cambio"]
                st.write(f"Cambio de puntaje: **{c['Puntaje']:+.1f}** · PD: **{c['PD']:+.2%}** · "
                         f"percentil: **{c['Percentil']:+.1f}** · pérdida esperada: **{c['Perdida_esperada_USD']:+,.0f} USD**")
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    partes = []
    for etiqueta, env in (("Reporte técnico", "ME_PRESTAS_REPORT_URL"), ("Video promocional", "ME_PRESTAS_VIDEO_URL"),
                          ("Repositorio", "ME_PRESTAS_REPO_URL")):
        url = os.environ.get(env, "").strip()
        partes.append(f"[{etiqueta}]({url})" if url else f"{etiqueta}: pendiente de publicar")
    st.markdown(" · ".join(partes))


if __name__ == "__main__":
    main()
