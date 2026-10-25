"""Calcula la pérdida esperada EL = PD * LGD * EAD. Responsable: P1."""
from src.utils.cli import build_parser, init_run


def run(cfg: dict) -> None:
    """Contrato:
    - Lee las predicciones de PD y LGD y la exposición (EAD) del conjunto de prueba.
    - Calcula EL = PD * LGD * EAD, vectorizado.
    - Escribe results/tables/expected_loss.csv.
    """
    raise NotImplementedError("TODO (P1): combinar PD, LGD y exposición")


def main():
    args = build_parser("Calcula la pérdida esperada", "configs/el.yaml").parse_args()
    cfg = init_run(args)
    print("[expected_loss] config:", cfg)
    # TODO (P1): run(cfg)


if __name__ == "__main__":
    main()