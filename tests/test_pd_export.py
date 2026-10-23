
import numpy as np

from src.models.mlp import MLP
from src.models.pd_mlp import predict_pd
from src.models.pd_export import save_pd_model, load_pd_model


def test_exportar_y_cargar_pd(tmp_path):
    config = {
        "input_dim": 10,
        "hidden_layers": (32, 16),
        "activation": "relu",
        "dropout": 0.2,
    }

    model = MLP(**config)
    model.eval()

    X = np.random.default_rng(42).normal(
        size=(20, 10)
    ).astype(np.float32)

    original = predict_pd(model, X)

    path = tmp_path / "pd_model.pt"

    save_pd_model(model, config, path)
    restored = load_pd_model(path)

    recovered = predict_pd(restored, X)

    np.testing.assert_allclose(
        original,
        recovered,
        rtol=1e-6,
        atol=1e-7,
    )
