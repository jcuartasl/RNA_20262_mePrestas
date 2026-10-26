"""Transformación de probabilidad de incumplimiento (PD) a puntaje de scorecard. Responsable: P4.

Convención: odds de BUEN pagador, odds = (1 - p) / p. Si z = logit(p), ln(odds) = -z.
    Factor = PDO / ln 2
    Offset = puntaje_base - Factor * ln(odds_base)
    Puntaje = Offset + Factor * ln(odds) = Offset - Factor * z
Mayor puntaje = menor riesgo. Duplicar las odds suma PDO puntos.
"""
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class Scorecard:
    puntaje_base: float = 600.0
    odds_base: float = 50.0     # odds de buen pagador en el puntaje base (50:1)
    pdo: float = 20.0           # puntos para duplicar las odds

    def __post_init__(self):
        if self.odds_base <= 0 or self.pdo <= 0:
            raise ValueError("odds_base y pdo deben ser positivos")

    @classmethod
    def desde_config(cls, cfg):
        return cls(float(cfg["base_score"]), float(cfg["base_odds"]), float(cfg["pdo"]))

    @property
    def factor(self):
        return self.pdo / np.log(2.0)

    @property
    def offset(self):
        return self.puntaje_base - self.factor * np.log(self.odds_base)

    def puntaje_desde_logit(self, z):
        """z = logit de la PD (salida de la red antes de la sigmoide, o decision_function de la logística)."""
        return self.offset - self.factor * np.asarray(z, dtype=float)

    def puntaje_desde_pd(self, p):
        """Preferir puntaje_desde_logit: con PD muy cercanas a 0 o 1 la PD pierde precisión numérica."""
        p = np.clip(np.asarray(p, dtype=float), np.finfo(float).tiny, np.nextafter(1.0, 0.0))
        return self.offset + self.factor * (np.log1p(-p) - np.log(p))

    def pd_desde_puntaje(self, puntaje):
        z = (self.offset - np.asarray(puntaje, dtype=float)) / self.factor
        return 1.0 / (1.0 + np.exp(-z))


def percentil(puntaje, referencia):
    """Porcentaje de la distribución de referencia con puntaje estrictamente menor (0 a 100)."""
    ref = np.sort(np.asarray(referencia, dtype=float))
    if ref.size == 0:
        raise ValueError("la distribución de referencia está vacía")
    return 100.0 * np.searchsorted(ref, np.asarray(puntaje, dtype=float), side="left") / ref.size
