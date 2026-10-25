"""Optimizadores escritos a mano (sgd y adam) para verificar las ecuaciones del reporte."""
import torch


def paso_sgd(params, lr=1e-3):
    """Un paso de descenso de gradiente: theta <- theta - lr * g."""
    if lr <= 0: raise ValueError("lr debe ser positivo")
    with torch.no_grad():
        for p in params: p -= lr * p.grad


def iniciar_adam(params):
    """Estado inicial de adam (params debe ser una lista): t = 0 y momentos m, v en cero."""
    ceros = lambda: [torch.zeros_like(p) for p in params]
    return {"t": 0, "m": ceros(), "v": ceros()}


def paso_adam(params, estado, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    """Un paso de adam: promedios móviles de g y g^2, corrección de sesgo y paso adaptativo."""
    b1, b2 = betas
    if lr <= 0: raise ValueError("lr debe ser positivo")
    if not (0 <= b1 < 1 and 0 <= b2 < 1): raise ValueError("betas deben estar en [0, 1)")
    # t se incrementa antes de usarlo: en el primer paso vale 1 y 1 - beta^t nunca es 0
    estado["t"] += 1
    t = estado["t"]
    with torch.no_grad():
        for p, m, v in zip(params, estado["m"], estado["v"]):
            m.mul_(b1).add_((1 - b1) * p.grad)
            v.mul_(b2).add_((1 - b2) * p.grad ** 2)
            # m y v arrancan en 0 y quedan sesgados hacia 0 por un factor (1 - beta^t)
            m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
            # eps va fuera de la raíz, como en el artículo original y en pytorch
            p -= lr * m_hat / (v_hat.sqrt() + eps)
