"""Funciones de pérdida derivadas por máxima verosimilitud, compartidas por PD y LGD."""
import torch


def _reducir(perdida, reduction):
    if reduction == "mean": return perdida.mean()
    if reduction == "sum": return perdida.sum()
    if reduction == "none": return perdida
    raise ValueError(f"reduction no válida: {reduction}")


def _bce_estable(logits, target):
    # -[y log s(z) + (1-y) log(1-s(z))] se simplifica a softplus(z) - y*z; logaddexp(0, z) es softplus
    # sin overflow y, a diferencia de max(z,0) + log1p(exp(-|z|)), autograd da bien el gradiente en z=0
    return torch.logaddexp(torch.zeros_like(logits), logits) - target * logits


def bce_con_logits(logits, target, reduction="mean"):
    """Log-verosimilitud negativa de una Bernoulli con p = sigmoide(logits), etiquetas 0/1 (pd)."""
    if logits.shape != target.shape: raise ValueError("logits y target deben tener la misma forma")
    if not torch.all((target == 0) | (target == 1)): raise ValueError("bce requiere etiquetas 0 o 1")
    return _reducir(_bce_estable(logits, target), reduction)


def bce_fraccional_con_logits(logits, target, reduction="mean"):
    """Cuasi-verosimilitud de Bernoulli para target en [0, 1]; se minimiza en la media de y."""
    if logits.shape != target.shape: raise ValueError("logits y target deben tener la misma forma")
    if not torch.all((target >= 0) & (target <= 1)): raise ValueError("bce requiere target en [0, 1]")
    # misma fórmula que la bce, pero con y continua no es una verosimilitud: no integra a 1 en [0, 1]
    return _reducir(_bce_estable(logits, target), reduction)


def gradiente_bce(logits, target):
    """Gradiente analítico de la bce promedio (binaria o fraccional) respecto a z: (s - y) / n."""
    return (torch.sigmoid(logits) - target) / logits.numel()


def mse_con_logits(logits, target, reduction="mean"):
    """Log-verosimilitud negativa (salvo constantes) de una normal con media sigmoide(logits)."""
    if logits.shape != target.shape: raise ValueError("logits y target deben tener la misma forma")
    if not torch.all((target >= 0) & (target <= 1)): raise ValueError("mse requiere target en [0, 1]")
    # sin logaritmos no hay overflow; el riesgo aquí es que el gradiente se apague si la sigmoide satura
    perdida = (torch.sigmoid(logits) - target) ** 2
    return _reducir(perdida, reduction)


def gradiente_mse(logits, target):
    """Gradiente analítico de la mse promedio respecto a los logits: 2 (s - y) s (1 - s) / n."""
    s = torch.sigmoid(logits)
    return 2 * (s - target) * s * (1 - s) / logits.numel()


def beta_con_logits(logits, target, log_phi, eps=1e-4, reduction="mean"):
    """Log-verosimilitud negativa de una beta con media sigmoide(logits) y precisión exp(log_phi)."""
    if logits.shape != target.shape: raise ValueError("logits y target deben tener la misma forma")
    if not torch.all((target >= 0) & (target <= 1)): raise ValueError("beta requiere target en [0, 1]")
    # con y = 0 o 1 exactos log(y) o log(1-y) es -inf, por eso se recorta a [eps, 1 - eps]
    y = target.clamp(eps, 1 - eps)
    # 1 - mu se calcula como sigmoide(-z) para no perderlo al restar; el piso evita lgamma(0) = inf
    piso = torch.finfo(logits.dtype).tiny
    mu, uno_menos_mu = torch.sigmoid(logits).clamp(min=piso), torch.sigmoid(-logits).clamp(min=piso)
    phi = torch.exp(log_phi)
    a, b = mu * phi, uno_menos_mu * phi
    perdida = (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(phi)
               - (a - 1) * torch.log(y) - (b - 1) * torch.log1p(-y))
    return _reducir(perdida, reduction)


def gradiente_beta(logits, target, log_phi, eps=1e-4):
    """Gradientes analíticos de la nll beta promedio respecto a los logits y a log_phi."""
    y, n, psi = target.clamp(eps, 1 - eps), logits.numel(), torch.special.digamma
    mu, phi = torch.sigmoid(logits), torch.exp(log_phi)
    a, b = mu * phi, (1 - mu) * phi
    # la beta compara en escala logit: psi(a) - psi(b) es E[logit(Y)] y log(y / (1-y)) es logit(y)
    d_logits = phi * mu * (1 - mu) * (psi(a) - psi(b) - torch.log(y / (1 - y))) / n
    d_phi = -psi(phi) + mu * psi(a) + (1 - mu) * psi(b) - mu * torch.log(y) - (1 - mu) * torch.log1p(-y)
    # regla de la cadena con phi = exp(log_phi): d/d(log_phi) = phi * d/d(phi)
    d_log_phi = (phi * d_phi).sum().reshape(log_phi.shape) / n
    return d_logits, d_log_phi


def _pesos(modelo):
    # los sesgos no se penalizan: fijan el nivel base (p. ej. la lgd media) y no hay por qué encogerlos
    return [p for nombre, p in modelo.named_parameters() if not nombre.endswith("bias")]


def penalizacion_l2(modelo):
    """Penalización l2 = 0.5 * suma(w^2) sobre los pesos; su gradiente es w, como el weight_decay."""
    return 0.5 * sum((w ** 2).sum() for w in _pesos(modelo))


def penalizacion_l1(modelo):
    """Penalización l1 = suma(|w|) sobre los pesos; su subgradiente es signo(w)."""
    return sum(w.abs().sum() for w in _pesos(modelo))


PERDIDAS = {"bce": bce_fraccional_con_logits, "mse": mse_con_logits, "beta": beta_con_logits}


def get_loss(name):
    """Devuelve la función de pérdida registrada con ese nombre."""
    if name not in PERDIDAS: raise ValueError(f"pérdida no soportada: {name} (hay: {list(PERDIDAS)})")
    return PERDIDAS[name]
