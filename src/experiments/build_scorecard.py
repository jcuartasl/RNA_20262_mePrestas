"""Construye el scorecard a partir de la PD. Responsable: P4."""
from src.utils.cli import build_parser, init_run


def run(cfg: dict) -> None:
    """Contrato:
    - Lee los splits de cfg["paths"]["splits"] y el modelo base de PD de models/.
    - Convierte PD en puntaje (base_score, pdo, base_odds definidos en configs/scorecard.yaml).
    - Escribe la tabla en results/tables/scorecard.csv.
    """
    raise NotImplementedError("TODO (P4): binning, puntos por variable y tabla de scorecard")


def main():
    args = build_parser("Construye el scorecard", "configs/scorecard.yaml").parse_args()
    cfg = init_run(args)
    print("[build_scorecard] config:", cfg)
    # TODO (P4): run(cfg)


if __name__ == "__main__":
    main()