"""Fija todas las fuentes de aleatoriedad para que los experimentos sean reproducibles."""
import os
import random
import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija la semilla en Python, NumPy y en el framework de redes si está instalado."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass