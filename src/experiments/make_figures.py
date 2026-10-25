"""Genera las figuras del informe a partir de results/tables y models/. Responsable: P4."""
from src.utils.cli import build_parser, init_run


def run(cfg: dict) -> None:
    """Contrato:
    - Solo lee resultados ya calculados; no reentrena ningún modelo.
    - Guarda las figuras en results/figures/ con nombres estables (fig_roc.png, fig_calibracion.png, ...).
    """
    raise NotImplementedError("TODO (P4): ROC, calibración, distribución de LGD, comparación de modelos")


def main():
    args = build_parser("Genera las figuras del informe", "configs/default.yaml").parse_args()
    cfg = init_run(args)
    print("[make_figures] config:", cfg)
    # TODO (P4): run(cfg)


if __name__ == "__main__":
    main()