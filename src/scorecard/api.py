"""Interfaz de la scorecard para la aplicación web (P5). Responsable: P4.

Contrato (guía, sección 7): score(features) -> (puntaje, PD) y percentile(puntaje).

Uso desde la app:
    from src.scorecard.api import ScorecardAPI
    api = ScorecardAPI.cargar()                          # una vez, al iniciar la app
    api.categorias()                                     # valores válidos para los menús
    r = api.evaluar({"loan_amnt": 10000, "term": " 36 months", ...})   # datos del formulario
    r["puntaje"], r["pd"], r["percentil"]
    api.que_pasaria_si(datos, {"annual_inc": 80000})     # escenario "qué pasaría si"
"""
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from src.experiments.build_scorecard import cargar_modelo_pd
from src.scorecard.scorecard import Scorecard, percentil
from src.utils.config import ROOT


class ScorecardAPI:
    def __init__(self, scorecard, logit, preprocesador, logits_referencia):
        self.scorecard = scorecard
        self._logit = logit
        self.preprocesador = preprocesador
        self._referencia = np.sort(scorecard.puntaje_desde_logit(logits_referencia))
        self.variables_requeridas = [c for nombre, _, cols in preprocesador.transformers_
                                     if nombre != "remainder" for c in cols]

    @classmethod
    def cargar(cls, ruta_config="configs/scorecard.yaml", **cambios):
        """Carga modelo, preprocesador y referencia. cambios: p. ej. pdo=40, base_score=700."""
        with open(ROOT / ruta_config, encoding="utf-8") as f:
            cfg = {**yaml.safe_load(f), **cambios}
        logit, _, _ = cargar_modelo_pd(cfg)
        ref = np.load(ROOT / "models" / f"scorecard_referencia_{cfg['modelo']}.npz")
        # la referencia se guardó como puntajes con sus propios parámetros; se pasa a logits
        # para que el percentil no dependa del puntaje base ni del PDO actuales
        logits_ref = (float(ref["offset"]) - ref["puntajes"].astype(float)) / float(ref["factor"])
        return cls(Scorecard.desde_config(cfg), logit, joblib.load(ROOT / "models" / "preprocessor.pkl"),
                   logits_ref)

    # ------------------------------------------------------------ contrato
    def score(self, X):
        """X ya preprocesado (n x 82). Devuelve (puntaje, PD) como arreglos."""
        z = np.asarray(self._logit(np.asarray(X, dtype=np.float32)), dtype=float)
        return self.scorecard.puntaje_desde_logit(z), expit(z)

    def percentile(self, puntaje):
        """Porcentaje de la población de referencia (entrenamiento) con puntaje menor."""
        return percentil(puntaje, self._referencia)

    # ------------------------------------------------------------ datos crudos (formulario)
    def preprocesar(self, datos):
        """dict (una persona) o DataFrame con las variables originales -> matriz de 82 columnas."""
        df = pd.DataFrame([datos]) if isinstance(datos, dict) else pd.DataFrame(datos).copy()
        faltantes = [c for c in self.variables_requeridas if c not in df.columns]
        if faltantes:
            raise ValueError(f"faltan variables: {faltantes}")
        for c in self.preprocesador.feature_names_in_:
            if c not in df.columns:
                df[c] = np.nan  # columnas que el preprocesador descarta (issue_d, target_pd, ...)
        return self.preprocesador.transform(df[list(self.preprocesador.feature_names_in_)]).astype(np.float32)

    def evaluar(self, datos):
        puntaje, pd_ = self.score(self.preprocesar(datos))
        perc = self.percentile(puntaje)
        if isinstance(datos, dict):
            return {"puntaje": float(puntaje[0]), "pd": float(pd_[0]), "percentil": float(perc[0])}
        return pd.DataFrame({"puntaje": puntaje, "pd": pd_, "percentil": perc})

    def que_pasaria_si(self, datos, cambios):
        antes, despues = self.evaluar(datos), self.evaluar({**datos, **cambios})
        return {"antes": antes, "despues": despues,
                "cambio_puntaje": despues["puntaje"] - antes["puntaje"],
                "cambio_pd": despues["pd"] - antes["pd"]}

    def categorias(self):
        """Valores aceptados por cada variable categórica (para menús desplegables)."""
        salida = {}
        for _, pipe, cols in self.preprocesador.transformers_:
            pasos = pipe.named_steps.values() if hasattr(pipe, "named_steps") else [pipe]
            for paso in pasos:
                if hasattr(paso, "categories_"):
                    salida.update({c: [str(v) for v in cats] for c, cats in zip(cols, paso.categories_)})
        return salida
