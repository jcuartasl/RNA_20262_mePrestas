
import time

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _prepare_X(X):
    """Convierte las características a una matriz numérica."""
    X = np.asarray(X, dtype=np.float64)

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X debe ser una matriz 2D no vacía")

    if not np.isfinite(X).all():
        raise ValueError("X contiene NaN o valores infinitos")

    return X


def _prepare_y(y):
    """Valida los valores observados de LGD."""
    y = np.asarray(y, dtype=np.float64).ravel()

    if y.size == 0 or not np.isfinite(y).all():
        raise ValueError("LGD contiene valores inválidos")

    if np.any((y < 0) | (y > 1)):
        raise ValueError("LGD debe estar entre 0 y 1")

    return y


def predict_lgd_linear(model, X, clip=True):
    """Predice LGD, opcionalmente limitada a [0, 1]."""
    X = _prepare_X(X)
    predictions = model.predict(X)

    if clip:
        predictions = np.clip(predictions, 0.0, 1.0)

    return predictions


def train_lgd_linear(
    X_train,
    y_train,
    X_val,
    y_val,
):
    """
    Entrena una regresión lineal de referencia para LGD.

    Los datos deben llegar preprocesados desde P1.
    No utiliza los datos de prueba.
    """

    X_train = _prepare_X(X_train)
    X_val = _prepare_X(X_val)

    y_train = _prepare_y(y_train)
    y_val = _prepare_y(y_val)

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Train y val deben tener iguales columnas")

    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train no coinciden")

    if len(X_val) != len(y_val):
        raise ValueError("X_val e y_val no coinciden")

    start = time.perf_counter()

    model = LinearRegression()
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start

    # Predicciones originales y acotadas
    raw_predictions = predict_lgd_linear(
        model, X_val, clip=False
    )

    predictions = np.clip(raw_predictions, 0.0, 1.0)

    metrics = {
        "mae": mean_absolute_error(y_val, predictions),
        "rmse": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "mae_raw": mean_absolute_error(
            y_val, raw_predictions
        ),
        "rmse_raw": np.sqrt(
            mean_squared_error(y_val, raw_predictions)
        ),
        "parameters": X_train.shape[1] + 1,
        "time_seconds": elapsed,
        "clipped_predictions": int(
            np.count_nonzero(
                (raw_predictions < 0) |
                (raw_predictions > 1)
            )
        ),
    }

    return model, metrics
