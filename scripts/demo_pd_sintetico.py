
import numpy as np
import matplotlib.pyplot as plt

from src.models.pd_mlp import train_pd_mlp, predict_pd
from src.models.learning_curves import plot_learning_curves


def main():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(800, 10)).astype(np.float32)

    logits = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2]
    )

    y = (logits > 0).astype(np.float32)

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
        X[:600],
        y[:600],
        X[600:],
        y[600:],
        config,
    )

    probabilities = predict_pd(model, X[600:])
    predictions = (probabilities >= 0.5).astype(int)

    accuracy = np.mean(predictions == y[600:])

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Mejor época: {history['best_epoch']}")
    print(f"Épocas ejecutadas: {history['epochs_trained']}")
    print(f"Mejor pérdida: {history['best_val_loss']:.4f}")

    fig = plot_learning_curves(
        history,
        save_path="results/figures/pd_learning_curve_demo.png",
    )

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
