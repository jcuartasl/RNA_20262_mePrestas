"""Entrena la red de PD (clasificación). Responsable: P2."""
from src.utils.cli import build_parser, init_run


def train(cfg: dict, seed: int) -> dict:
    """Contrato: entrena con `seed` y devuelve {"metrics": {...}, "n_params": int}."""
    raise NotImplementedError("TODO (P2): entrenar y guardar el modelo en models/")


def main():
    args = build_parser("Entrena la red de PD", "configs/pd_mlp.yaml").parse_args()
    cfg = init_run(args)
    print("[train_pd] config:", cfg)
    # TODO (P2): llamar a train(cfg, cfg["seed"]) y registrar con log_run(...)


if __name__ == "__main__":
    main()