from src.utils.registry import log_run, summarize_budget

def main():
    cfg = {"model": "MLP", "hidden_layers": [64, 32], "lr": 0.01}
    log_run("test_experiment", cfg, seed=42, n_params=123456, seconds=12.34, metrics={"accuracy": 0.95})
    summary = summarize_budget("test_experiment")
    print(summary)

if __name__ == "__main__":
    main()