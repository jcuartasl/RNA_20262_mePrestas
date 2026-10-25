"""Funciones de pérdida derivadas por máxima verosimilitud, compartidas por PD y LGD."""
import torch


def _reducir(perdida, reduction):
    if reduction == "mean": return perdida.mean()
    if reduction == "sum": return perdida.sum()
    if reduction == "none": return perdida
    raise ValueError(f"reduction no válida: {reduction}")


def bce_con_logits(logits, target, reduction="mean"):
    """Log-verosimilitud negativa de una Bernoulli con p = sigmoide(logits)."""
    if logits.shape != target.shape: raise ValueError("logits y target deben tener la misma forma")
    if not torch.all((target == 0) | (target == 1)): raise ValueError("bce requiere etiquetas 0 o 1")
    # -[y log s(z) + (1-y) log(1-s(z))] se simplifica a softplus(z) - y*z; logaddexp(0, z) es softplus
    # sin overflow y, a diferencia de max(z,0) + log1p(exp(-|z|)), autograd da bien el gradiente en z=0
    perdida = torch.logaddexp(torch.zeros_like(logits), logits) - target * logits
    return _reducir(perdida, reduction)


def gradiente_bce(logits, target):
    """Gradiente analítico de la bce promedio respecto a los logits: (sigmoide(z) - y) / n."""
    return (torch.sigmoid(logits) - target) / logits.numel()


PERDIDAS = {"bce": bce_con_logits}


def get_loss(name):
    """Devuelve la función de pérdida registrada con ese nombre."""
    if name not in PERDIDAS: raise ValueError(f"pérdida no soportada: {name} (hay: {list(PERDIDAS)})")
    return PERDIDAS[name]
