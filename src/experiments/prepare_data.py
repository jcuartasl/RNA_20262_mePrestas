"""Limpia los datos, define los objetivos PD/LGD y genera los splits. Responsable: P1."""
from src.utils.cli import build_parser, init_run


def run(cfg: dict) -> None:
    """Contrato:
    - Lee cfg["paths"]["raw"].
    - Escribe en cfg["paths"]["processed"] y cfg["paths"]["splits"]
      (train / val / test, generados una sola vez con cfg["seed"]).
    - Columnas objetivo: target_pd (0/1) y target_lgd (0 a 1, solo incumplidos).
    """
    raise NotImplementedError("TODO (P1): limpieza, objetivos PD/LGD y splits")


def main():
    args = build_parser("Prepara datos y splits", "configs/default.yaml").parse_args()
    cfg = init_run(args)
    print("[prepare_data] config:", cfg)
    # TODO (P1): run(cfg)


if __name__ == "__main__":
    main()