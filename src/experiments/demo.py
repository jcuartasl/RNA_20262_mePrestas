"""Demo reducida (< 2 min): encadena los pasos con configs/demo.yaml. Responsable: P5."""
import importlib
import time

from src.utils.cli import build_parser, init_run

# (nombre, módulo, función). Los entrenamientos usan train(cfg, seed); el resto, run(cfg).
STEPS = [
    ("prepare_data", "src.experiments.prepare_data", "run"),
    ("train_pd", "src.experiments.train_pd", "train"),
    ("train_lgd", "src.experiments.train_lgd", "train"),
    ("expected_loss", "src.experiments.expected_loss", "run"),
    ("build_scorecard", "src.experiments.build_scorecard", "run"),
    ("make_figures", "src.experiments.make_figures", "run"),
]


def main():
    args = build_parser("Demo reducida de todo el flujo", "configs/demo.yaml").parse_args()
    cfg = init_run(args)
    print("[demo] config:", cfg)

    t0 = time.perf_counter()
    for name, module, func in STEPS:
        fn = getattr(importlib.import_module(module), func)
        try:
            fn(cfg, cfg["seed"]) if func == "train" else fn(cfg)
            print(f"[demo] {name}: ok")
        except NotImplementedError:
            print(f"[demo] {name}: pendiente (stub)")
    print(f"[demo] total: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()