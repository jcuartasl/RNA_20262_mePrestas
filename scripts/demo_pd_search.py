
import numpy as np

from src.search.pd_random_search import random_search_pd


def main():

    rng = np.random.default_rng(42)

    # Datos sintéticos equivalentes a la demostración anterior
    X = rng.normal(
        size=(800, 10)
    ).astype(np.float32)

    logits = (
        2.0 * X[:, 0]
        - 1.5 * X[:, 1]
        + 0.8 * X[:, 2]
    )

    y = (logits > 0).astype(np.float32)

    # Misma partición para todas las configuraciones
    X_train, X_val = X[:600], X[600:]
    y_train, y_val = y[:600], y[600:]

    best_config, results = random_search_pd(
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials=8,
        seed=42,
    )

    print("\n===== MEJOR CONFIGURACIÓN =====")

    for key, value in best_config.items():
        print(f"{key}: {value}")

    print("\n===== MEJORES 3 EXPERIMENTOS =====")

    for result in results[:3]:
        print(
            f"Experimento {result['trial']}: "
            f"BCE={result['val_loss']:.5f}, "
            f"capas={result['hidden_layers']}, "
            f"lr={result['lr']}"
        )


if __name__ == "__main__":
    main()
