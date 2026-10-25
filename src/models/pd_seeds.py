
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

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
    experiment_name="pd_temporal",
):
    """Entrena un MLP de PD con distintas semillas."""

    if n_seeds < 1:
        raise ValueError("n_seeds debe ser positivo")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    y_val = np.asarray(y_val).ravel()

    if len(np.unique(y_val)) != 2:
        raise ValueError(
            "La validación debe contener ambas clases para AUC"
        )

    results = []
    predictions = {
        "y_true": y_val.astype(int),
    }

    total_start = time.perf_counter()

    for seed in range(n_seeds):
        current_config = {**config, "seed": seed}

        start = time.perf_counter()

        model, history = train_pd_mlp(
            X_train,
            y_train,
            X_val,
            y_val,
            current_config,
        )

        probabilities = predict_pd(model, X_val)
        elapsed = time.perf_counter() - start

        auc = float(
            roc_auc_score(y_val, probabilities)
        )

        brier = float(
            brier_score_loss(y_val, probabilities)
        )

        bce = float(
            log_loss(y_val, probabilities, labels=[0, 1])
        )

        row = {
            "seed": seed,
            "auc": auc,
            "gini": 2 * auc - 1,
            "brier": brier,
            "bce": bce,
            "best_epoch": history["best_epoch"],
            "epochs_trained": history["epochs_trained"],
            "parameters": count_parameters(model),
            "time_seconds": round(elapsed, 3),
        }

        results.append(row)
        predictions[f"seed_{seed}"] = probabilities

        print(
            f"Semilla {seed + 1}/{n_seeds} | "
            f"AUC={auc:.5f} | "
            f"Brier={brier:.5f} | "
            f"BCE={bce:.5f} | "
            f"Tiempo={elapsed:.1f}s",
            flush=True,
        )

        # Guardar avance después de cada semilla
        result_path = output / f"{experiment_name}_seeds.csv"

        with result_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(row.keys()),
            )
            writer.writeheader()
            writer.writerows(results)

        # Predicciones para P4
        pred_path = output / f"{experiment_name}_predictions.csv"

        with pred_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            columns = list(predictions.keys())
            writer.writerow(columns)
            writer.writerows(
                zip(*(predictions[c] for c in columns))
            )

    # Resumen estadístico
    summary = {}

    for metric in ("auc", "gini", "brier", "bce"):
        values = np.asarray(
            [row[metric] for row in results],
            dtype=float,
        )

        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1))
            if len(values) > 1 else None,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    summary["n_seeds"] = n_seeds
    summary["parameters"] = results[0]["parameters"]
    summary["total_time_seconds"] = round(
        time.perf_counter() - total_start, 3
    )
    summary["config"] = {
        **config,
        "hidden_layers": list(config["hidden_layers"]),
    }

    summary_path = output / f"{experiment_name}_summary.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n===== RESUMEN FINAL =====")

    for metric in ("auc", "gini", "brier", "bce"):
        s = summary[metric]
        print(
            f"{metric.upper()}: "
            f"{s['mean']:.5f} ± {s['std']:.5f}"
        )

    print(f"\nResultados: {result_path}")
    print(f"Predicciones: {pred_path}")
    print(f"Resumen: {summary_path}")

    return results, summary
