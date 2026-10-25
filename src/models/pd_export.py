
from pathlib import Path

import torch

from src.models.mlp import build_mlp


def save_pd_model(model, config, path):
    """Guarda arquitectura y pesos entrenados de PD."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_config = {
        "input_dim": int(config["input_dim"]),
        "hidden_layers": list(config["hidden_layers"]),
        "activation": config["activation"],
        "dropout": float(config["dropout"]),
        "output_activation": "none",
    }

    checkpoint = {
        "architecture": model_config,
        "state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
    }

    torch.save(checkpoint, path)


def load_pd_model(path, device="cpu"):
    """Reconstruye el MLP y carga sus pesos."""

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model = build_mlp(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    return model
