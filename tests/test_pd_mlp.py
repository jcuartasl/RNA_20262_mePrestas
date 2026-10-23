
import numpy as np
import torch

from src.models.pd_mlp import train_pd_mlp, predict_pd


def test_entrenamiento_pd():
    """Comprueba el entrenador usando datos sintéticos."""

    rng = np.random.default_rng(42)

    # 800 clientes ficticios y 10 características
    X = rng.normal(size=(800, 10)).astype(np.float32)

    # Relación artificial para construir las etiquetas
    logits = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2]
    )

    # Etiquetas binarias: 0 o 1
    y = (logits > 0).astype(np.float32)

    # Separación: 600 entrenamiento, 200 validación
    X_train, X_val = X[:600], X[600:]
    y_train, y_val = y[:600], y[600:]

    config = {
        "seed": 42,
        "hidden_layers": [16, 8],
        "activation": "relu",
        "dropout": 0.0,
        "lr": 0.01,
        "epochs": 80,
        "batch_size": 64,
        "patience": 12,
        "l2": 0.0001,
    }

    model, history = train_pd_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        config,
    )

    probabilities = predict_pd(model, X_val)

    predictions = (probabilities >= 0.5).astype(int)
    accuracy = np.mean(predictions == y_val)

    assert len(probabilities) == len(y_val)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.isfinite(history["best_val_loss"])
    assert 1 <= history["best_epoch"] <= config["epochs"]
    assert accuracy >= 0.85

    print(f"\nAccuracy de validación: {accuracy:.3f}")
    print(f"Mejor época: {history['best_epoch']}")
    print(f"Mejor pérdida: {history['best_val_loss']:.4f}")


def test_reproducibilidad_pd():
    """Comprueba dos entrenamientos con la misma semilla."""

    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 4)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

    config = {
        "seed": 42,
        "hidden_layers": [8],
        "dropout": 0.0,
        "epochs": 5,
        "batch_size": 16,
        "patience": 3,
    }

    model_a, _ = train_pd_mlp(
        X[:80], y[:80], X[80:], y[80:], config
    )

    model_b, _ = train_pd_mlp(
        X[:80], y[:80], X[80:], y[80:], config
    )

    predictions_a = predict_pd(model_a, X[80:])
    predictions_b = predict_pd(model_b, X[80:])

    np.testing.assert_allclose(
        predictions_a,
        predictions_b,
        rtol=1e-5,
        atol=1e-6,
    )
