"""Entrenamiento oficial del clasificador PD. Responsable: P2.

Cambios respecto a la versión anterior:
  * La demo (sample_fraction < 1) entrena con una muestra del train y guarda el modelo con otro
    nombre (pd_mlp_demo_<esquema>_seed<semilla>.pt): ya no pisa el modelo real.
  * La demo se registra como experimento "pd_mlp_demo", así no contamina el presupuesto de "pd_mlp".
  * La semilla es obligatoria (cfg["seed"]): no hay valor por defecto silencioso (antes 42).
  * El modelo se guarda en ROOT/models (antes "models" relativo al directorio de ejecución).
  * Se añade KS a las métricas (el enunciado exige AUC, Gini y KS).
"""
import time

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.data.loader import load_splits
from src.evaluation.metrics import ks as ks_stat
from src.models.pd_export import save_pd_model
from src.models.pd_mlp import predict_pd, train_pd_mlp
from src.utils.cli import build_parser, init_run
from src.utils.config import ROOT
from src.utils.registry import log_run


def _muestra(X, y, fraccion: float, seed: int):
    """Submuestra reproducible del train para la demo. Con fraccion >= 1 devuelve todo."""
    if fraccion >= 1.0:
        return X, y
    n = max(2, int(len(y) * fraccion))
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    return X[idx], y[idx]


def ruta_modelo(scheme: str, seed: int, demo: bool):
    """models/pd_mlp_<esquema>_seed<s>.pt (real) o models/pd_mlp_demo_<esquema>_seed<s>.pt (demo)."""
    prefijo = "pd_mlp_demo" if demo else "pd_mlp"
    return ROOT / "models" / f"{prefijo}_{scheme}_seed{int(seed)}.pt"


def train(cfg: dict, seed: int) -> dict:
    """
    Entrena el MLP de PD usando la configuración oficial.

    Contrato con run_seeds:
      {"metrics": {...}, "n_params": int, "model_path": str}
    """
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})

    # Clasificación binaria: BCE, no MSE
    loss_name = str(loss_cfg.get("name", "bce")).lower()
    if loss_name not in ("bce", "bcewithlogits"):
        raise ValueError("El clasificador PD requiere BCEWithLogitsLoss")

    scheme = data_cfg.get("split", cfg.get("scheme", "temporal"))
    if scheme not in ("temporal", "random"):
        raise ValueError("Partición no válida")

    fraccion = float(cfg.get("sample_fraction", 1.0))
    if not 0.0 < fraccion <= 1.0:
        raise ValueError("sample_fraction debe estar en (0, 1]")
    demo = fraccion < 1.0

    data = load_splits("pd", scheme)
    # la demo recorta solo el train; validación queda completa (igual que train_lgd)
    X_train, y_train = _muestra(data["X_train"], data["y_train"], fraccion, int(seed))

    model_config = {
        "seed": int(seed),
        "hidden_layers": tuple(model_cfg.get("hidden_layers", model_cfg.get("hidden", [64, 32]))),
        "activation": model_cfg.get("activation", "relu"),
        "dropout": float(model_cfg.get("dropout", 0.0)),
        "lr": float(train_cfg.get("lr", 0.003)),
        "epochs": int(train_cfg.get("epochs", 30)),
        "batch_size": int(train_cfg.get("batch_size", 512)),
        "patience": int(train_cfg.get("patience", 5)),
        "l2": float(train_cfg.get("l2", model_cfg.get("l2", 0.0001))),
    }

    model, history = train_pd_mlp(X_train, y_train, data["X_val"], data["y_val"], model_config)

    probabilities = predict_pd(model, data["X_val"])
    y_val = data["y_val"]

    auc = float(roc_auc_score(y_val, probabilities))
    metrics = {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(ks_stat(y_val, probabilities)),
        "brier": float(brier_score_loss(y_val, probabilities)),
        "bce": float(history["best_val_loss"]),
        "best_epoch": int(history["best_epoch"]),
        "epochs_trained": int(history["epochs_trained"]),
    }

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_path = ruta_modelo(scheme, seed, demo)
    save_pd_model(
        model,
        {
            "input_dim": int(X_train.shape[1]),
            "hidden_layers": model_config["hidden_layers"],
            "activation": model_config["activation"],
            "dropout": model_config["dropout"],
        },
        model_path,
    )

    return {"metrics": metrics, "n_params": n_params, "model_path": str(model_path)}


def main():
    parser = build_parser("Entrena la red de PD", "configs/pd_mlp.yaml")
    args = parser.parse_args()
    cfg = init_run(args)

    seed = int(cfg["seed"])          # obligatorio: default.yaml la define (90)
    demo = float(cfg.get("sample_fraction", 1.0)) < 1.0

    start = time.perf_counter()
    result = train(cfg, seed)
    elapsed = time.perf_counter() - start

    # la demo no se cuenta en el presupuesto de búsqueda de "pd_mlp"
    log_run("pd_mlp_demo" if demo else "pd_mlp", cfg, seed, result["n_params"], elapsed, result["metrics"])

    print("\n===== ENTRENAMIENTO PD" + (" (DEMO, muestra del train)" if demo else "") + " =====")
    print("Métricas:", result["metrics"])
    print("Parámetros:", result["n_params"])
    print("Modelo guardado:", result["model_path"])


if __name__ == "__main__":
    main()