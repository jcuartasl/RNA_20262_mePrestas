"""Corre un modelo con las 30 semillas y registra cada corrida."""
import importlib
import time

from src.utils.cli import build_parser, init_run
from src.utils.registry import log_run
from src.utils.seed import set_seed

# modelo -> (módulo, config por defecto)
MODELS = {
    "pd_mlp": ("src.experiments.train_pd", "configs/pd_mlp.yaml"),
    "lgd_mlp": ("src.experiments.train_lgd", "configs/lgd_mlp.yaml"),
}


def main():
    p = build_parser("Corre un modelo con varias semillas")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--n_seeds", type=int, default=30)
    args = p.parse_args()
    args.config = args.config or MODELS[args.model][1]

    cfg = init_run(args)
    train = importlib.import_module(MODELS[args.model][0]).train

    for seed in cfg["run_seeds"][: args.n_seeds]:
        set_seed(seed)
        t0 = time.perf_counter()
        out = train(cfg, seed)
        log_run(args.model, cfg, seed, out["n_params"],
                time.perf_counter() - t0, out["metrics"])
        print(f"[run_seeds] {args.model} seed={seed} {out['metrics']}")


if __name__ == "__main__":
    main()