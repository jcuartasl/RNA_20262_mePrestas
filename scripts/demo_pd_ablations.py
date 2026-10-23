
import numpy as np

from src.search.pd_ablations import run_pd_ablations


def main():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(800, 10)).astype(np.float32)

    logits = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2]
    )

    y = (logits > 0).astype(np.float32)

    run_pd_ablations(
        X[:600],
        y[:600],
        X[600:],
        y[600:],
    )


if __name__ == "__main__":
    main()
