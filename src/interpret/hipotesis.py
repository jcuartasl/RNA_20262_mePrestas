"""Pruebas estadísticas para el contraste de hipótesis del EDA. Responsable: P4."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm, ttest_ind


def comparar_tasas(y_a, y_b):
    """Tasa de incumplimiento del grupo A (riesgo esperado mayor) frente a B.

    Prueba z de dos proporciones, unilateral (H1: tasa_a > tasa_b); IC 95 % de la razón por log-RR.
    """
    a, b = np.asarray(y_a, dtype=float), np.asarray(y_b, dtype=float)
    na, nb, pa, pb = a.size, b.size, a.mean(), b.mean()
    p = (a.sum() + b.sum()) / (na + nb)
    z = (pa - pb) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    se_log = np.sqrt((1 - pa) / (pa * na) + (1 - pb) / (pb * nb))
    razon = pa / pb
    return {"n_a": na, "n_b": nb, "valor_a": pa, "valor_b": pb, "diferencia": pa - pb, "razon": razon,
            "ic_inf": razon * np.exp(-1.96 * se_log), "ic_sup": razon * np.exp(1.96 * se_log),
            "p_valor": float(norm.sf(z))}


def comparar_medias(x_a, x_b):
    """Media del grupo A frente a B: t de Welch unilateral (H1: media_a > media_b); IC 95 % de la diferencia."""
    a, b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    dif = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return {"n_a": a.size, "n_b": b.size, "valor_a": a.mean(), "valor_b": b.mean(), "diferencia": dif,
            "razon": a.mean() / b.mean(), "ic_inf": dif - 1.96 * se, "ic_sup": dif + 1.96 * se,
            "p_valor": float(ttest_ind(a, b, equal_var=False, alternative="greater").pvalue)}


def _logverosimilitud_logistica(Z, y):
    """Máxima verosimilitud de una logística (con intercepto) por BFGS con gradiente analítico."""
    Z = np.column_stack([np.ones(len(y)), Z])

    def nll(b):
        eta = Z @ b
        return np.sum(np.logaddexp(0, eta) - y * eta)

    def grad(b):
        return Z.T @ (1 / (1 + np.exp(-(Z @ b))) - y)
    res = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="BFGS")
    return -res.fun


def prueba_no_linealidad(x, y):
    """Razón de verosimilitud: logit lineal en x frente a x + x^2 + x^3 (modelos anidados, 2 g.l.)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = (x - x.mean()) / x.std()
    ll_lineal = _logverosimilitud_logistica(x[:, None], y)
    ll_cubico = _logverosimilitud_logistica(np.column_stack([x, x ** 2, x ** 3]), y)
    lr = max(2 * (ll_cubico - ll_lineal), 0.0)
    return {"lr": lr, "p_valor": float(chi2.sf(lr, df=2))}
