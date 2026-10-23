
import torch


def test_gradiente_bce_con_autograd():
    """Compara el gradiente analítico con PyTorch."""

    z = torch.tensor(
        [0.5, -1.0, 2.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    y = torch.tensor(
        [1.0, 0.0, 1.0],
        dtype=torch.float64,
    )

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        z, y, reduction="mean"
    )

    loss.backward()

    # El promedio divide el gradiente entre n.
    analytical = (torch.sigmoid(z.detach()) - y) / len(y)

    torch.testing.assert_close(
        z.grad,
        analytical,
        rtol=1e-10,
        atol=1e-12,
    )


def test_gradiente_con_diferencias_finitas():
    """Compara autograd contra aproximación numérica."""

    z = torch.tensor(
        0.7,
        dtype=torch.float64,
        requires_grad=True,
    )

    y = torch.tensor(1.0, dtype=torch.float64)

    def loss_fn(value):
        return torch.nn.functional.binary_cross_entropy_with_logits(
            value, y
        )

    loss_fn(z).backward()
    automatic = z.grad.item()

    h = 1e-5

    with torch.no_grad():
        numerical = (
            loss_fn(z + h).item()
            - loss_fn(z - h).item()
        ) / (2 * h)

    assert abs(automatic - numerical) < 1e-7
