"""Fija todas las fuentes de aleatoriedad para que los experimentos sean reproducibles."""
import os
import random
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # para reproducibilidad en GPU (PyTorch)

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija la semilla en Python, NumPy y en el framework de redes si está instalado."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass