
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP


def set_seed(seed: int = 42):
    """Establece semillas para reproducir experimentos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pd_mlp(
    X_train,
    y_train,
    X_val,
    y_val,
    config=None,
):
    """
    Entrena un MLP para clasificación binaria PD.

    Recibe datos ya preprocesados y separados.
    Devuelve el modelo y el historial de entrenamiento.
    """

    config = config or {}

    seed = int(config.get("seed", 42))
    set_seed(seed)

    lr = float(config.get("lr", 0.001))
    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 128))
    patience = int(config.get("patience", 10))
    l2 = float(config.get("l2", 0.0))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Convertir los datos a tensores
    X_train = torch.as_tensor(
        np.asarray(X_train, dtype=np.float32)
    )
    X_val = torch.as_tensor(
        np.asarray(X_val, dtype=np.float32)
    )

    y_train = torch.as_tensor(
        np.asarray(y_train, dtype=np.float32)
    ).reshape(-1, 1)

    y_val = torch.as_tensor(
        np.asarray(y_val, dtype=np.float32)
    ).reshape(-1, 1)

    # Verificaciones básicas
    if X_train.ndim != 2 or X_val.ndim != 2:
        raise ValueError("X debe ser una matriz 2D")

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Las entradas deben tener iguales columnas")

    if (
        len(X_train) == 0
        or len(X_val) == 0
        or len(X_train) != len(y_train)
        or len(X_val) != len(y_val)
    ):
        raise ValueError("Dimensiones o tamaños inválidos")

    if not (
        torch.all((y_train == 0) | (y_train == 1))
        and torch.all((y_val == 0) | (y_val == 1))
    ):
        raise ValueError("PD requiere etiquetas 0 o 1")

    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size y patience deben ser positivos")

    # Crear modelo utilizando nuestro MLP compartido
    model = MLP(
        input_dim=X_train.shape[1],
        hidden_layers=tuple(
            config.get("hidden_layers", [64, 32])
        ),
        activation=config.get("activation", "relu"),
        dropout=float(config.get("dropout", 0.2)),
        output_activation="none",
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=l2,
    )

    dataset = TensorDataset(X_train, y_train)

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    # Entrenamiento por épocas
    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)

        train_loss = total_loss / len(dataset)

        # Validación sin actualizar los pesos
        model.eval()

        with torch.no_grad():
            val_logits = model(X_val.to(device))
            val_loss = criterion(
                val_logits,
                y_val.to(device)
            ).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Guardar la mejor versión del modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    # Recuperar pesos de la mejor época
    model.load_state_dict(best_weights)
    model.eval()

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val_loss
    history["epochs_trained"] = len(history["train_loss"])

    return model, history


@torch.no_grad()
def predict_pd(model, X):
    """Devuelve probabilidades de incumplimiento."""
    device = next(model.parameters()).device

    X = torch.as_tensor(
        np.asarray(X, dtype=np.float32)
    ).to(device)

    model.eval()
    logits = model(X)
    probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy().ravel()
