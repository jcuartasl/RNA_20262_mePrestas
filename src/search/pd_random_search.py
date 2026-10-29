
import csv
import itertools
import random
import time
from pathlib import Path

from src.models.pd_mlp import train_pd_mlp


def count_parameters(model):
    """Cuenta los parámetros entrenables del MLP."""
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def random_search_pd(
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=8,
    seed=90,
    output_path="results/tables/pd_search_demo.csv",
):
    """
    Busca hiperparámetros del MLP de PD.

    Selecciona la configuración con menor BCE
    de validación.
    """

    if n_trials < 1:
        raise ValueError("n_trials debe ser positivo")

    # Espacio de búsqueda inicial
    search_space = {
        "hidden_layers": [
            (16,),
            (32, 16),
            (64, 32),
        ],
        "activation": ["relu", "tanh"],
        "dropout": [0.0, 0.2, 0.3],
        "lr": [0.001, 0.003, 0.01],
    }

    keys = list(search_space)

    # Generar todas las combinaciones posibles
    combinations = list(
        itertools.product(
            *(search_space[key] for key in keys)
        )
    )

    rng = random.Random(seed)
    rng.shuffle(combinations)

    selected = combinations[:min(n_trials, len(combinations))]

    results = []
    best_config = None
    best_loss = float("inf")

    total_start = time.perf_counter()

    for trial_number, combination in enumerate(selected, start=1):

        params = dict(zip(keys, combination))

        config = {
            **params,
            "seed": seed,
            "epochs": 60,
            "batch_size": 64,
            "patience": 10,
            "l2": 0.0001,
        }

        print(
            f"\nExperimento {trial_number}/{len(selected)}"
        )
        print(f"Configuración: {config}")

        start = time.perf_counter()

        model, history = train_pd_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            config,
        )

        elapsed = time.perf_counter() - start

        val_loss = history["best_val_loss"]

        result = {
            "trial": trial_number,
            "hidden_layers": str(params["hidden_layers"]),
            "activation": params["activation"],
            "dropout": params["dropout"],
            "lr": params["lr"],
            "val_loss": val_loss,
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "parameters": count_parameters(model),
            "time_seconds": round(elapsed, 3),
        }

        results.append(result)

        print(f"BCE validación: {val_loss:.5f}")
        print(f"Tiempo: {elapsed:.2f} segundos")

        if val_loss < best_loss:
            best_loss = val_loss
            best_config = config.copy()

    # Ordenar del menor al mayor error
    results.sort(key=lambda r: r["val_loss"])

    # Guardar resultados en CSV
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start

    print("\n===== RESULTADOS DE LA BÚSQUEDA =====")
    print(f"Configuraciones evaluadas: {len(results)}")
    print(f"Tiempo total: {total_time:.2f} segundos")
    print(f"Mejor BCE de validación: {best_loss:.5f}")
    print(f"Mejor configuración: {best_config}")
    print(f"Resultados guardados en: {path}")

    return best_config, results
