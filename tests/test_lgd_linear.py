
import numpy as np
import pytest

from src.models.lgd_linear import (
    train_lgd_linear,
    predict_lgd_linear,
)


def test_entrenamiento_lgd():
    """Comprueba que el modelo aprende una relación lineal."""

    rng = np.random.default_rng(42)

    X = rng.uniform(-1, 1, size=(500, 5))

    # LGD sintética con relación lineal conocida
    y = (
        0.5
        + 0.15 * X[:, 0]
        - 0.10 * X[:, 1]
        + 0.05 * X[:, 2]
    )

    model, metrics = train_lgd_linear(
        X[:400],
        y[:400],
        X[400:],
        y[400:],
    )

    predictions = predict_lgd_linear(model, X[400:])

    assert len(predictions) == 100
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 1)

    assert metrics["mae"] < 0.01
    assert metrics["rmse"] < 0.01
    assert metrics["parameters"] == 6

    print(f"\nMAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")


def test_predicciones_acotadas():
    """Comprueba el recorte a [0, 1]."""

    X_train = np.array([
        [0.0],
        [0.2],
        [0.4],
        [0.6],
    ])

    y_train = np.array([
        0.0, 0.2, 0.4, 0.6
    ])

    model, _ = train_lgd_linear(
        X_train,
        y_train,
        X_train,
        y_train,
    )

    X_extreme = np.array([
        [-10.0],
        [10.0],
    ])

    raw = predict_lgd_linear(
        model, X_extreme, clip=False
    )

    clipped = predict_lgd_linear(
        model, X_extreme, clip=True
    )

    assert np.any((raw < 0) | (raw > 1))
    assert np.all((clipped >= 0) & (clipped <= 1))


def test_lgd_fuera_de_rango():
    """Rechaza objetivos LGD inválidos."""

    X = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    y = np.array([
        0.2, 0.5, 1.2
    ])

    with pytest.raises(ValueError):
        train_lgd_linear(X, y, X, y)
