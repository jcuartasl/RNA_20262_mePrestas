
import torch

from src.models.mlp import MLP, build_mlp


def test_mlp_pd():
    """Verifica que el MLP funcione para clasificación PD."""

    model = MLP(
        input_dim=10,
        hidden_layers=(64, 32),
        activation="relu",
        dropout=0.2,
        output_activation="none",
    )

    X = torch.randn(20, 10)

    logits = model(X)
    probabilities = torch.sigmoid(logits)

    assert logits.shape == (20, 1)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)


def test_mlp_lgd():
    """Verifica que el MLP pueda producir valores entre 0 y 1."""

    cfg = {
        "input_dim": 10,
        "hidden_layers": [32, 16],
        "activation": "relu",
        "dropout": 0.1,
        "output_activation": "sigmoid",
    }

    model = build_mlp(cfg)

    X = torch.randn(20, 10)

    predictions = model(X)

    assert predictions.shape == (20, 1)
    assert torch.all(predictions >= 0)
    assert torch.all(predictions <= 1)


def test_backward():
    """Verifica que PyTorch calcule los gradientes."""

    model = MLP(input_dim=10)

    X = torch.randn(20, 10)
    y = torch.randint(0, 2, (20, 1)).float()

    criterion = torch.nn.BCEWithLogitsLoss()

    logits = model(X)
    loss = criterion(logits, y)

    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_mlp_aprende_xor():
    """Comprueba que el MLP pueda aprender XOR."""

    torch.manual_seed(42)

    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])

    y = torch.tensor([
        [0.],
        [1.],
        [1.],
        [0.]
    ])

    model = MLP(
        input_dim=2,
        hidden_layers=(16, 16),
        activation="tanh",
        dropout=0.0,
        output_activation="none"
    )

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.03
    )

    # Entrenamiento
    model.train()

    for epoch in range(2000):
        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

    # Evaluación
    model.eval()

    with torch.no_grad():
        probabilities = torch.sigmoid(model(X))
        predictions = (probabilities >= 0.5).float()

    assert torch.equal(predictions, y), (
        f"Predicciones incorrectas: {probabilities}"
    )
