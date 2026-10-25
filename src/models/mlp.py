
import torch
from torch import nn


class MLP(nn.Module):
    """
    Perceptrón multicapa configurable para PD y LGD.

    La salida por defecto son logits para clasificación PD.
    Para LGD puede configurarse una salida sigmoide.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: tuple = (64, 32),
        activation: str = "relu",
        dropout: float = 0.2,
        output_activation: str = "none",
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim debe ser positivo")

        if not 0 <= dropout < 1:
            raise ValueError("dropout debe estar entre 0 y 1")

        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }

        if activation not in activations:
            raise ValueError(
                f"Activación no soportada: {activation}"
            )

        if output_activation not in ("none", "sigmoid"):
            raise ValueError("Activación de salida no soportada")

        layers = []
        previous_dim = input_dim

        # Construcción de las capas ocultas
        for neurons in hidden_layers:

            if neurons <= 0:
                raise ValueError(
                    "Cada capa debe tener al menos una neurona"
                )

            layers.append(nn.Linear(previous_dim, neurons))
            layers.append(activations[activation]())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            previous_dim = neurons

        # Capa de salida: una única predicción
        layers.append(nn.Linear(previous_dim, 1))

        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_mlp(cfg: dict):
    """Construye el MLP desde un diccionario de configuración."""
    return MLP(
        input_dim=cfg["input_dim"],
        hidden_layers=tuple(cfg.get("hidden_layers", [64, 32])),
        activation=cfg.get("activation", "relu"),
        dropout=cfg.get("dropout", 0.2),
        output_activation=cfg.get("output_activation", "none"),
    )
