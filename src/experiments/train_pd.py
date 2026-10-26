
"""Entrenamiento oficial del clasificador PD. Responsable: P2."""

import time
from pathlib import Path

from sklearn.metrics import roc_auc_score, brier_score_loss

from src.data.loader import load_splits
from src.models.pd_mlp import train_pd_mlp, predict_pd
from src.models.pd_export import save_pd_model
from src.utils.cli import build_parser, init_run
from src.utils.registry import log_run


def train(cfg: dict, seed: int) -> dict:
    """
    Entrena el MLP de PD usando la configuración oficial.

    Contrato con run_seeds:
      {
        "metrics": {...},
        "n_params": int
      }
    """
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    # Clasificación binaria: BCE, no MSE
    loss_name = loss_cfg.get("name", "bce").lower()

    if loss_name not in ("bce", "bcewithlogits"):
        raise ValueError(
            "El clasificador PD requiere BCEWithLogitsLoss"
        )

    scheme = data_cfg.get(
        "split",
        cfg.get("scheme", "temporal"),
    )

    if scheme not in ("temporal", "random"):
        raise ValueError("Partición no válida")

    data = load_splits("pd", scheme)

    model_config = {
        "seed": int(seed),
        "hidden_layers": tuple(
            model_cfg.get(
                "hidden_layers",
                model_cfg.get("hidden", [64, 32]),
            )
        ),
        "activation": model_cfg.get(
            "activation", "relu"
        ),
        "dropout": float(
            model_cfg.get("dropout", 0.0)
        ),
        "lr": float(train_cfg.get("lr", 0.003)),
        "epochs": int(train_cfg.get("epochs", 30)),
        "batch_size": int(
            train_cfg.get("batch_size", 512)
        ),
        "patience": int(
            train_cfg.get("patience", 5)
        ),
        "l2": float(
            train_cfg.get(
                "l2",
                model_cfg.get("l2", 0.0001),
            )
        ),
    }

    model, history = train_pd_mlp(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        model_config,
    )

    probabilities = predict_pd(
        model,
        data["X_val"],
    )

    y_val = data["y_val"]

    auc = float(
        roc_auc_score(y_val, probabilities)
    )

    brier = float(
        brier_score_loss(y_val, probabilities)
    )

    metrics = {
        "auc": auc,
        "gini": 2 * auc - 1,
        "brier": brier,
        "bce": float(history["best_val_loss"]),
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(
            history["epochs_trained"]
        ),
    }

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    # Exportar con nombre diferente por semilla
    output = Path("models")
    output.mkdir(parents=True, exist_ok=True)

    model_path = (
        output / f"pd_mlp_{scheme}_seed{seed}.pt"
    )

    save_pd_model(
        model,
        {
            "input_dim": int(data["X_train"].shape[1]),
            "hidden_layers": model_config["hidden_layers"],
            "activation": model_config["activation"],
            "dropout": model_config["dropout"],
        },
        model_path,
    )

    return {
        "metrics": metrics,
        "n_params": n_params,
        "model_path": str(model_path),
    }


def main():
    parser = build_parser(
        "Entrena la red de PD",
        "configs/pd_mlp.yaml",
    )
    args = parser.parse_args()
    cfg = init_run(args)

    seed = int(cfg.get("seed", 42))

    start = time.perf_counter()
    result = train(cfg, seed)
    elapsed = time.perf_counter() - start

    log_run(
        "pd_mlp",
        cfg,
        seed,
        result["n_params"],
        elapsed,
        result["metrics"],
    )

    print("\n===== ENTRENAMIENTO PD =====")
    print("Métricas:", result["metrics"])
    print("Parámetros:", result["n_params"])
    print("Modelo guardado:", result["model_path"])


if __name__ == "__main__":
    main()
