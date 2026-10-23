
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

from src.models.pd_mlp import train_pd_mlp, predict_pd
from src.search.pd_random_search import count_parameters


def run_pd_seeds(
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    n_seeds=30,
    output_dir="results/tables",
):
    """Ejecuta el MLP de PD con diferentes semillas."""

    if n_seeds < 1:
        raise ValueError("n_seeds debe ser positivo")

    y_val = np.asarray(y_val).ravel()

    results = []
    start_total = time.perf_counter()

    for seed in range(n_seeds):
        current_config = {**config, "seed": seed}

        start = time.perf_counter()

        model, history = train_pd_mlp(
            X_train, y_train,
            X_val, y_val,
            current_config,
        )

        probabilities = predict_pd(model, X_val)

        elapsed = time.perf_counter() - start

        # AUC no se define si solo hay una clase.
        auc = (
            roc_auc_score(y_val, probabilities)
            if len(np.unique(y_val)) == 2
            else float("nan")
        )

        result = {
            "seed": seed,
            "auc": float(auc),
            "brier": float(
                brier_score_loss(y_val, probabilities)
            ),
            "bce": float(history["best_val_loss"]),
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "parameters": count_parameters(model),
            "time_seconds": round(elapsed, 4),
        }

        results.append(result)

        print(
            f"Semilla {seed + 1}/{n_seeds} | "
            f"AUC={auc:.4f} | "
            f"Brier={result['brier']:.4f} | "
            f"BCE={result['bce']:.4f}"
        )

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "pd_seeds_demo.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    summary = {}

    for metric in ("auc", "brier", "bce"):
        values = np.asarray(
            [row[metric] for row in results],
            dtype=float,
        )
        valid = values[np.isfinite(values)]

        summary[metric] = {
            "mean": float(np.mean(valid)) if valid.size else None,
            "std": float(np.std(valid, ddof=1))
                   if valid.size > 1 else None,
            "min": float(np.min(valid)) if valid.size else None,
            "max": float(np.max(valid)) if valid.size else None,
        }

    summary["n_seeds"] = n_seeds
    summary["total_time_seconds"] = round(
        time.perf_counter() - start_total, 4
    )
    summary["parameters"] = results[0]["parameters"]
    summary["config"] = config

    with (path / "pd_seeds_summary_demo.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    return results, summary
