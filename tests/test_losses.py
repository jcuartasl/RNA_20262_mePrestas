"""Pruebas de src/models/losses.py: valores de referencia, gradientes y estabilidad numérica."""
import pytest
import torch
import torch.nn.functional as F

from src.models.losses import bce_con_logits, get_loss, gradiente_bce, gradiente_mse, mse_con_logits


def datos_bce(n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(n, generator=g, dtype=torch.float64)
    target = torch.randint(0, 2, (n,), generator=g).double()
    return logits, target


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_bce_contra_pytorch(reduction):
    logits, target = datos_bce()
    esperado = F.binary_cross_entropy_with_logits(logits, target, reduction=reduction)
    torch.testing.assert_close(bce_con_logits(logits, target, reduction), esperado)


def test_bce_valor_en_cero():
    # con z=0 el modelo dice 50 %, así que la pérdida es -log(0.5) = log 2 sea y 0 o 1
    logits, target = torch.zeros(2, dtype=torch.float64), torch.tensor([0.0, 1.0], dtype=torch.float64)
    log2 = torch.tensor(2.0, dtype=torch.float64).log()
    torch.testing.assert_close(bce_con_logits(logits, target), log2)


def test_gradiente_contra_autograd():
    logits, target = datos_bce()
    # se agregan dos z=0 exactos porque ahí es donde una forma estable mal escrita rompe el gradiente
    logits = torch.cat([logits, torch.zeros(2, dtype=torch.float64)]).requires_grad_()
    target = torch.cat([target, torch.tensor([0.0, 1.0], dtype=torch.float64)])
    (gradiente,) = torch.autograd.grad(bce_con_logits(logits, target), logits)
    torch.testing.assert_close(gradiente, gradiente_bce(logits.detach(), target))


def test_gradcheck_diferencias_finitas():
    logits, target = datos_bce(n=10)
    assert torch.autograd.gradcheck(lambda z: bce_con_logits(z, target), (logits.requires_grad_(),))


@pytest.mark.parametrize("z", [50.0, 100.0, -50.0, -100.0])
def test_bce_estable_con_logits_extremos(z):
    # float32 a propósito, que es donde aparece el overflow
    logits, target = torch.full((2,), z, requires_grad=True), torch.tensor([0.0, 1.0])
    perdida = bce_con_logits(logits, target, reduction="none")
    perdida.sum().backward()
    assert torch.isfinite(perdida).all() and torch.isfinite(logits.grad).all()
    # el ejemplo que el modelo falla paga |z| y el que acierta paga casi 0
    fallado, acertado = (perdida[0], perdida[1]) if z > 0 else (perdida[1], perdida[0])
    assert fallado.item() == pytest.approx(abs(z)) and acertado.item() == pytest.approx(0.0, abs=1e-6)


def test_bce_ingenua_falla_con_logits_extremos():
    # por esto existe la forma estable: en float32 sigmoide(-100) da 0 y log(0) = -inf
    logits, target = torch.tensor([-100.0]), torch.tensor([1.0])
    sigmoide = torch.sigmoid(logits)
    ingenua = -(target * torch.log(sigmoide) + (1 - target) * torch.log(1 - sigmoide))
    assert torch.isinf(ingenua).all() and torch.isfinite(bce_con_logits(logits, target))


def test_bce_rechaza_entradas_invalidas():
    with pytest.raises(ValueError): bce_con_logits(torch.zeros(3), torch.zeros(3, 1))
    with pytest.raises(ValueError): bce_con_logits(torch.zeros(2), torch.tensor([0.0, 0.5]))
    with pytest.raises(ValueError): bce_con_logits(torch.zeros(2), torch.zeros(2), reduction="media")


def datos_lgd(n=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = 3 * torch.randn(n, generator=g, dtype=torch.float64)
    target = torch.rand(n, generator=g, dtype=torch.float64)
    target[:2] = torch.tensor([0.0, 1.0], dtype=torch.float64)  # la lgd real también trae 0 y 1 exactos
    return logits, target


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_mse_contra_pytorch(reduction):
    logits, target = datos_lgd()
    esperado = F.mse_loss(torch.sigmoid(logits), target, reduction=reduction)
    torch.testing.assert_close(mse_con_logits(logits, target, reduction), esperado)


def test_mse_valor_a_mano():
    # z=0 predice 0.5 y con y=0.7 el error cuadrado es 0.2^2 = 0.04
    logits, target = torch.zeros(1, dtype=torch.float64), torch.tensor([0.7], dtype=torch.float64)
    torch.testing.assert_close(mse_con_logits(logits, target), torch.tensor(0.04, dtype=torch.float64))


def test_gradiente_mse_contra_autograd():
    logits, target = datos_lgd()
    logits.requires_grad_()
    (gradiente,) = torch.autograd.grad(mse_con_logits(logits, target), logits)
    torch.testing.assert_close(gradiente, gradiente_mse(logits.detach(), target))


def test_gradcheck_mse():
    logits, target = datos_lgd(n=10)
    assert torch.autograd.gradcheck(lambda z: mse_con_logits(z, target), (logits.requires_grad_(),))


@pytest.mark.parametrize("z", [50.0, -50.0])
def test_mse_se_satura_con_logits_extremos(z):
    # la red está muy segura y equivocada (predice ~1 o ~0 con y=0.5) y aun así casi no se corrige,
    # porque el factor s(1-s) de la regla de la cadena se va a 0; con bce el gradiente sería ~0.5
    logits, target = torch.tensor([z], requires_grad=True), torch.tensor([0.5])
    perdida = mse_con_logits(logits, target)
    perdida.backward()
    assert torch.isfinite(perdida) and perdida.item() == pytest.approx(0.25)
    assert logits.grad.abs().item() < 1e-10


def test_mse_rechaza_entradas_invalidas():
    with pytest.raises(ValueError): mse_con_logits(torch.zeros(3), torch.zeros(3, 1))
    with pytest.raises(ValueError): mse_con_logits(torch.zeros(2), torch.tensor([0.5, 1.5]))
    with pytest.raises(ValueError): mse_con_logits(torch.zeros(2), torch.tensor([-0.1, 0.5]))


def test_get_loss():
    assert get_loss("bce") is bce_con_logits
    assert get_loss("mse") is mse_con_logits
    with pytest.raises(ValueError): get_loss("hinge")
