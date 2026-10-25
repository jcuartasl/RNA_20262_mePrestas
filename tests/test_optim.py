"""Pruebas de src/models/optim.py: adam propio contra pytorch y propiedades de sgd y adam."""
import copy

import pytest
import torch

from src.models.losses import get_loss, penalizacion_l2
from src.models.mlp import build_mlp
from src.models.optim import iniciar_adam, paso_adam, paso_sgd


def logistica_de_juguete(seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(200, 5, generator=g, dtype=torch.float64)
    ruido = 0.3 * torch.randn(200, generator=g, dtype=torch.float64)
    y = (X @ torch.tensor([1.0, -2.0, 0.5, 0.0, 3.0], dtype=torch.float64) + ruido > 0).double()
    params = [0.1 * torch.randn(5, generator=g, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)]
    return X, y, [p.requires_grad_() for p in params]


def calcular_gradiente(params, X, y, escala=1.0):
    for p in params: p.grad = None
    w, b = params
    (escala * get_loss("bce")(X @ w + b, y)).backward()


def test_adam_igual_a_pytorch():
    X, y, propios = logistica_de_juguete()
    referencia = [p.detach().clone().requires_grad_() for p in propios]
    estado, optimizador = iniciar_adam(propios), torch.optim.Adam(referencia, lr=0.05)
    for _ in range(50):
        calcular_gradiente(propios, X, y)
        paso_adam(propios, estado, lr=0.05)
        calcular_gradiente(referencia, X, y)
        optimizador.step()
        # se compara en cada paso y no solo al final, para que un error no se compense después
        for a, b in zip(propios, referencia): torch.testing.assert_close(a, b)


def test_adam_primer_paso_mide_lr():
    # en t=1, m_hat = g y v_hat = g^2, así que el paso es lr * g / |g| tanto si g = 0.004 como si g = 100
    p = torch.tensor([3.0, -0.002, 50.0], dtype=torch.float64, requires_grad=True)
    inicial = p.detach().clone()
    (p ** 2).sum().backward()
    paso_adam([p], iniciar_adam([p]), lr=0.01)
    torch.testing.assert_close(inicial - p.detach(), 0.01 * torch.sign(inicial), rtol=0, atol=1e-7)
    # sin la corrección de sesgo el primer paso sería 0.1 g / sqrt(0.001 g^2) = sqrt(10) ~ 3.16 veces lr
    g = 2 * inicial
    sin_correccion = 0.01 * (0.1 * g) / ((0.001 * g ** 2).sqrt() + 1e-8)
    assert (sin_correccion.abs() / 0.01).tolist() == pytest.approx([10 ** 0.5] * 3, rel=1e-3)


def test_adam_invariante_a_la_escala_y_sgd_no():
    def correr(paso, escala):
        X, y, params = logistica_de_juguete()
        estado = iniciar_adam(params)
        for _ in range(20):
            calcular_gradiente(params, X, y, escala)
            paso(params, estado)
        return [p.detach() for p in params]

    adam = lambda params, estado: paso_adam(params, estado, lr=0.05)
    sgd = lambda params, estado: paso_sgd(params, lr=1e-4)
    for a, b in zip(correr(adam, 1.0), correr(adam, 1000.0)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
    # con sgd multiplicar la pérdida por 1000 multiplica cada paso por 1000
    X, y, params = logistica_de_juguete()
    inicial = [p.detach().clone() for p in params]
    desplazamiento = [(a - b).abs().max() for a, b in zip(correr(sgd, 1000.0), inicial)]
    desplazamiento_base = [(a - b).abs().max() for a, b in zip(correr(sgd, 1.0), inicial)]
    assert all(d > 100 * d0 for d, d0 in zip(desplazamiento, desplazamiento_base))


def test_cuadratica_mal_condicionada_sgd_diverge_y_adam_converge():
    # f = (100 x^2 + y^2) / 2: con lr > 2/100 sgd se pasa de largo en x y diverge; adam reescala cada eje
    def correr(paso):
        p = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        estado = iniciar_adam([p])
        for _ in range(2000):
            p.grad = None
            (0.5 * (100 * p[0] ** 2 + p[1] ** 2)).backward()
            paso([p], estado)
        return p.detach()

    sgd = correr(lambda params, estado: paso_sgd(params, lr=0.025))
    adam = correr(lambda params, estado: paso_adam(params, estado, lr=0.025))
    assert not torch.isfinite(sgd).all() or sgd.abs().max() > 1e6
    assert adam.abs().max() < 1e-6


def grupos_sin_sesgos(modelo, lam):
    pesos = [p for n, p in modelo.named_parameters() if not n.endswith("bias")]
    sesgos = [p for n, p in modelo.named_parameters() if n.endswith("bias")]
    return [{"params": pesos, "weight_decay": lam}, {"params": sesgos, "weight_decay": 0.0}]


def test_weight_decay_de_adam_es_l2_y_adamw_no():
    # evidencia del hallazgo h1: Adam(weight_decay) suma lam*w al gradiente, que es justo el gradiente
    # de lam * 0.5 * |w|^2; AdamW resta lam*w aparte, sin pasar por m y v, así que es otra cosa
    X, y, _ = logistica_de_juguete()
    torch.manual_seed(0)
    base = build_mlp({"input_dim": 5, "hidden_layers": [8], "dropout": 0.0}).double()
    con_wd, con_l2, con_adamw = (copy.deepcopy(base) for _ in range(3))
    lam = 0.1
    optimizadores = [torch.optim.Adam(grupos_sin_sesgos(con_wd, lam), lr=0.01),
                     torch.optim.Adam(con_l2.parameters(), lr=0.01),
                     torch.optim.AdamW(grupos_sin_sesgos(con_adamw, lam), lr=0.01)]
    for _ in range(30):
        for modelo, optimizador in zip((con_wd, con_l2, con_adamw), optimizadores):
            optimizador.zero_grad()
            perdida = get_loss("bce")(modelo(X).squeeze(1), y)
            if modelo is con_l2: perdida = perdida + lam * penalizacion_l2(modelo)
            perdida.backward()
            optimizador.step()
        for a, b in zip(con_wd.parameters(), con_l2.parameters()): torch.testing.assert_close(a, b)
    pares = zip(con_adamw.parameters(), con_l2.parameters())
    assert max((a - b).abs().max().item() for a, b in pares) > 1e-3


def test_l2_encoge_los_pesos():
    # una logística es un mlp sin capas ocultas; mayor lam = a priori más estrecha = pesos más chicos
    X, y, _ = logistica_de_juguete()
    normas = []
    for lam in [0.0, 0.01, 0.1, 1.0]:
        torch.manual_seed(0)
        modelo = build_mlp({"input_dim": 5, "hidden_layers": [], "dropout": 0.0}).double()
        params = list(modelo.parameters())
        estado = iniciar_adam(params)
        for _ in range(300):
            modelo.zero_grad()
            (get_loss("bce")(modelo(X).squeeze(1), y) + lam * penalizacion_l2(modelo)).backward()
            paso_adam(params, estado, lr=0.05)
        normas.append(modelo.network[0].weight.norm().item())
    assert all(a > b for a, b in zip(normas, normas[1:]))


def test_optimizadores_rechazan_parametros_invalidos():
    p = torch.zeros(2, requires_grad=True)
    with pytest.raises(ValueError): paso_sgd([p], lr=0)
    with pytest.raises(ValueError): paso_adam([p], iniciar_adam([p]), lr=-1)
    with pytest.raises(ValueError): paso_adam([p], iniciar_adam([p]), betas=(0.9, 1.0))
