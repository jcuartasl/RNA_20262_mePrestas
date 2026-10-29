"""Streamlit app for the ¿Me prestas? credit-risk prototype.

Copy to app/app.py in the repository and run from the repository root:
    streamlit run app/app.py
Loads saved models; it does not re-train them.
"""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import streamlit as st
import yaml
import sys
from pathlib import Path
from src.models.lgd_mlp import cargar_lgd_mlp, predecir_lgd
from src.scorecard.api import ScorecardAPI
from src.utils.config import ROOT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="¿Me prestas? · Riesgo de crédito", page_icon="📊", layout="wide")

@st.cache_resource(show_spinner="Cargando modelos guardados…")
def cargar_recursos():
    api = ScorecardAPI.cargar()
    with (ROOT / "configs" / "lgd_mlp.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    seed = int(os.environ.get("ME_PRESTAS_LGD_SEED"))
    scheme = cfg.get("data", {}).get("scheme", "random")
    suffix = "" if scheme == "random" else f"_{scheme}"
    model_path = ROOT / "models" / f"lgd_mlp{suffix}_seed{seed}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No existe el modelo LGD: {model_path.relative_to(ROOT)}")
    lgd_model, _, loss_name = cargar_lgd_mlp(model_path)
    return api, lgd_model, loss_name, model_path, scheme


def detalles_preprocesador(api):
    numeric, initial = [], {}
    for name, transformer, columns in api.preprocesador.transformers_:
        if name == "remainder" or not isinstance(columns, (list, tuple, np.ndarray)):
            continue
        columns = list(columns)
        imputer = transformer.named_steps.get("imputer") if hasattr(transformer, "named_steps") else None
        if name == "num":
            numeric.extend(columns)
            if imputer is not None:
                initial.update({c: float(v) for c, v in zip(columns, imputer.statistics_)})
        elif name == "cat" and imputer is not None:
            initial.update({c: str(v) for c, v in zip(columns, imputer.statistics_)})
    return numeric, initial


def agrupar(columns):
    groups = {"Solicitud y monto": [], "Ingresos y empleo": [], "Historial de crédito": [], "Otros datos": []}
    patterns = {
        "Solicitud y monto": ("loan_amnt", "funded_amnt", "funded_amnt_inv", "term", "purpose"),
        "Ingresos y empleo": ("annual_inc", "emp_length", "home_ownership", "verification_status"),
        "Historial de crédito": ("delinq", "inq_", "open_acc", "pub_rec", "revol", "total_acc", "earliest_cr_line", "collections", "acc_", "mortgage", "bankcard", "num_", "tax_liens", "months_since"),
    }
    for col in columns:
        group = next((g for g, keys in patterns.items() if any(k in col.lower() for k in keys)), "Otros datos")
        groups[group].append(col)
    return {g: cols for g, cols in groups.items() if cols}


def widget(col, numeric, initial, categories, key):
    label = col.replace("_", " ").capitalize()
    if numeric:
        value = float(initial.get(col, 0.0))
        return st.number_input(label, value=value, step=1.0 if value.is_integer() else 0.1, key=key,
                               help=f"Valor inicial: mediana del entrenamiento ({value:,.3g}).")
    options = categories.get(col, [])
    if not options:
        return st.text_input(label, value=str(initial.get(col, "")), key=key)
    default = str(initial.get(col, options[0]))
    return st.selectbox(label, options, index=options.index(default) if default in options else 0, key=key)


def predict(api, lgd_model, features: dict[str, Any], ead: float):
    X = api.preprocesar(features)
    score, pd_value = api.score(X)
    lgd = float(predecir_lgd(lgd_model, X)[0])
    pd_value, score = float(pd_value[0]), float(score[0])
    return {"PD": pd_value, "Puntaje": score, "Percentil": float(api.percentile([score])[0]),
            "LGD": lgd, "EAD": float(ead), "E[L]": pd_value * lgd * float(ead)}


def show_metrics(result, title):
    st.subheader(title)
    cols = st.columns(5)
    for col, label, value in zip(cols, ("PD", "Puntaje", "Percentil", "LGD", "Pérdida esperada"),
                                 (f"{result['PD']:.1%}", f"{result['Puntaje']:.1f}", f"{result['Percentil']:.1f}",
                                  f"{result['LGD']:.1%}", f"${result['E[L]']:,.2f}")):
        col.metric(label, value)
    st.caption(f"E[L] = PD × LGD × EAD = {result['PD']:.4f} × {result['LGD']:.4f} × ${result['EAD']:,.2f}.")


def main():
    st.title("¿Me prestas?")
    st.write("Explora una estimación de riesgo crediticio con los modelos guardados del proyecto.")
    st.info("Prototipo académico; no constituye una decisión real de crédito. Consulta el reporte para conocer la definición de LGD y las limitaciones temporales del modelo.")
    try:
        api, lgd_model, loss_name, lgd_path, scheme = cargar_recursos()
    except Exception as exc:
        st.error(f"No se pudieron cargar los artefactos del proyecto: {exc}")
        st.stop()

    numeric, initial = detalles_preprocesador(api)
    categories = api.categorias()
    features_required = api.variables_requeridas
    st.caption(f"Scorecard: base {api.scorecard.puntaje_base:g}, PDO {api.scorecard.pdo:g}. LGD: {loss_name}, split {scheme}, `{lgd_path.relative_to(ROOT)}`.")

    with st.form("credit_case"):
        st.subheader("Características del solicitante y del crédito")
        st.caption("Los valores iniciales corresponden a medianas/modas del entrenamiento; ajústalos al escenario que quieras estimar.")
        ead_default = max(1.0, float(initial.get("funded_amnt", 10000.0)))
        ead = st.number_input("EAD / monto financiado (USD)", min_value=0.01, value=ead_default, step=100.0)
        features = {}
        for title, fields in agrupar(features_required).items():
            with st.expander(title, expanded=(title == "Solicitud y monto")):
                cols = st.columns(2)
                for i, field in enumerate(fields):
                    with cols[i % 2]:
                        features[field] = widget(field, field in numeric, initial, categories, f"base_{field}")
        submitted = st.form_submit_button("Calcular estimación", type="primary", use_container_width=True)

    if submitted:
        # En la definición adoptada por el proyecto EAD = funded_amnt al originar.
        if "funded_amnt" in features:
            features["funded_amnt"] = float(ead)
        try:
            st.session_state["base_features"] = features
            st.session_state["base_ead"] = float(ead)
            st.session_state["base_result"] = predict(api, lgd_model, features, float(ead))
        except Exception as exc:
            st.error(f"No se pudo calcular el caso; revisa los datos ingresados. Detalle: {exc}")

    base = st.session_state.get("base_result")
    base_features = st.session_state.get("base_features")
    if base and base_features:
        show_metrics(base, "Estimación del caso base")
        st.divider()
        st.subheader("¿Qué pasaría si cambias una característica?")
        candidates = [c for c in features_required if c != "funded_amnt"]
        if candidates:
            variable = st.selectbox("Característica", candidates, key="scenario_variable")
            if variable in numeric:
                current = float(base_features[variable])
                changed_value = st.number_input(f"Nuevo valor de {variable.replace('_', ' ')}", value=current,
                                                step=1.0 if current.is_integer() else 0.1, key="scenario_numeric")
            else:
                options = categories.get(variable, [])
                current = str(base_features[variable])
                index = options.index(current) if current in options else 0
                changed_value = st.selectbox(f"Nuevo valor de {variable.replace('_', ' ')}", options or [current],
                                              index=index, key="scenario_category")
            if st.button("Comparar escenario"):
                scenario = {**base_features, variable: changed_value}
                try:
                    alternative = predict(api, lgd_model, scenario, st.session_state["base_ead"])
                    left, right = st.columns(2)
                    with left:
                        show_metrics(base, "Caso base")
                    with right:
                        show_metrics(alternative, "Escenario modificado")
                    st.write(f"Cambio de puntaje: **{alternative['Puntaje'] - base['Puntaje']:+.1f}** · "
                             f"Cambio de PD: **{alternative['PD'] - base['PD']:+.1%}** · "
                             f"Cambio de pérdida esperada: **${alternative['E[L]'] - base['E[L]']:+,.2f}**")
                except Exception as exc:
                    st.error(f"No se pudo calcular el escenario: {exc}")

    links = []
    for label, env_name in (("Reporte técnico", "ME_PRESTAS_REPORT_URL"), ("Video promocional", "ME_PRESTAS_VIDEO_URL")):
        url = os.environ.get(env_name, "").strip()
        if url:
            links.append(f"[{label}]({url})")
    st.divider()
    st.markdown(" · ".join(links) if links else "Configura `ME_PRESTAS_REPORT_URL` y `ME_PRESTAS_VIDEO_URL` para mostrar los enlaces finales.")


if __name__ == "__main__":
    main()
