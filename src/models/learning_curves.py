
from pathlib import Path

import matplotlib.pyplot as plt


def plot_learning_curves(history, save_path=None):
    """Grafica las pérdidas de entrenamiento y validación."""

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        epochs,
        train_loss,
        label="Entrenamiento",
        linewidth=2,
    )

    ax.plot(
        epochs,
        val_loss,
        label="Validación",
        linewidth=2,
    )

    ax.axvline(
        history["best_epoch"],
        linestyle="--",
        label=f"Mejor época: {history['best_epoch']}",
    )

    ax.set_title("Curvas de aprendizaje - MLP de PD")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida BCE")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)

    return fig
