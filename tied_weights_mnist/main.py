from pathlib import Path

from config import Config
from data import load_mnist


def main():
    config = Config()
    train_dataloader, test_dataloader = load_mnist(
        batch_size=config.batch_size, data_dir=Path(__file__).parent / config.data_dir
    )


if __name__ == "__main__":
    main()
