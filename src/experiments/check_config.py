from src.utils.config import load_config


def main():
    base = load_config("configs/default.yaml")
    nuevo = load_config("configs/default.yaml", ["train=0.2", "seed=42"])
    print("Base:", base)
    print("Nuevo:", nuevo)


if __name__ == "__main__":
    main()