"""Entrena la red de LGD (regresión). Responsable: P3."""
from src.utils.cli import build_parser, init_run


def train(cfg: dict, seed: int) -> dict:
    """Contrato: entrena con `seed` y devuelve {"metrics": {...}, "n_params": int}.

    - Entrena solo con los incumplidos (target_lgd definido).
    - Guarda el modelo con la semilla en el nombre: models/lgd_mlp_seed{seed}.pt
    - Claves de metrics acordadas: rmse, mae, r2.
    """
    raise NotImplementedError("TODO (P3): entrenar la red de LGD y guardar el modelo en models/")


def main():
    args = build_parser("Entrena la red de LGD", "configs/lgd_mlp.yaml").parse_args()
    cfg = init_run(args)
    print("[train_lgd] config:", cfg)
    # TODO (P3): llamar a train(cfg, cfg["seed"]) y registrar con log_run(...)


if __name__ == "__main__":
    main()