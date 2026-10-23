
import numpy as np

from src.models.pd_seeds import run_pd_seeds


def main():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(800, 10)).astype(np.float32)

    z = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2]
    y = (z > 0).astype(np.float32)

    config = {
        "hidden_layers": (32, 16),
        "activation": "tanh",
        "dropout": 0.0,
        "lr": 0.003,
        "epochs": 60,
        "batch_size": 64,
        "patience": 10,
        "l2": 0.0001,
    }

    results, summary = run_pd_seeds(
        X[:600],
        y[:600],
        X[600:],
        y[600:],
        config=config,
        n_seeds=3,
    )

    print("\nRESUMEN")
    for metric in ("auc", "brier", "bce"):
        print(metric, summary[metric])


if __name__ == "__main__":
    main()
