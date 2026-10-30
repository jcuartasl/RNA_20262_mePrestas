
import csv
import time
from pathlib import Path

from src.models.pd_mlp import train_pd_mlp
from src.search.pd_random_search import count_parameters


def run_pd_ablations(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path="results/tables/pd_ablations_demo.csv",
):
    """Evalúa modificaciones controladas del MLP."""

    base_config = {
        "hidden_layers": (32, 16),
        "activation": "tanh",
        "dropout": 0.2,
        "lr": 0.003,
        "seed": 90,
        "epochs": 60,
        "batch_size": 64,
        "patience": 10,
        "l2": 0.0001,
    }

    experiments = [
        ("Referencia", {}),
        ("Sin dropout", {"dropout": 0.0}),
        ("Dropout 0.3", {"dropout": 0.3}),
        ("Menor profundidad", {"hidden_layers": (32,)}),
        ("Mayor profundidad", {"hidden_layers": (32, 16, 8)}),
        ("Mayor ancho", {"hidden_layers": (64, 32)}),
    ]

    results = []

    for name, changes in experiments:
        config = {**base_config, **changes}

        start = time.perf_counter()

        model, history = train_pd_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            config,
        )

        elapsed = time.perf_counter() - start

        result = {
            "experiment": name,
            "hidden_layers": str(config["hidden_layers"]),
            "activation": config["activation"],
            "dropout": config["dropout"],
            "lr": config["lr"],
            "val_loss": history["best_val_loss"],
            "best_epoch": history["best_epoch"],
            "parameters": count_parameters(model),
            "time_seconds": round(elapsed, 3),
        }

        results.append(result)

        print(
            f"{name}: "
            f"BCE={result['val_loss']:.5f}, "
            f"parámetros={result['parameters']}"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResultados guardados en: {path}")

    return results
