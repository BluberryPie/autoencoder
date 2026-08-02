import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config
from data import load_mnist
from model import AutoEncoder, TiedAutoEncoder
from visualize import plot_loss_curves


ModelType = AutoEncoder | TiedAutoEncoder


def train(
    config: Config,
    model: ModelType,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module = nn.BCELoss(),
) -> tuple[float, float]:
    train_losses, eval_losses = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        train_loss = 0.0
        for X, _ in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = loss_fn(preds, X)
            train_loss += loss.detach() * X.size(0)
            loss.backward()
            optimizer.step()
        train_loss = train_loss.item() / len(train_loader.dataset)
        eval_loss = evaluate(model, test_loader) / len(test_loader.dataset)
        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
        logging.info(
            f"[Epoch {epoch + 1:>2}/{config.num_epochs}] => "
            f"Train Loss: {train_loss:.4f} / Eval Loss: {eval_loss:.4f}"
        )
    return train_losses, eval_losses


@torch.no_grad()
def evaluate(
    model: ModelType, test_loader: DataLoader, loss_fn: nn.Module = nn.BCELoss()
) -> float:
    model.eval()  # Switch model to eval mode
    total_loss = 0.0
    for X, _ in test_loader:
        preds = model(X)
        total_loss += loss_fn(preds, X).detach() * X.size(0)
    model.train()  # Switch model back to training mode
    return total_loss.item()


def main():
    # Initialize logger / configurations
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    config = Config()

    # Initialize data loaders
    train_loader, test_loader = load_mnist(
        batch_size=config.batch_size, data_dir=Path(__file__).parent / config.data_dir
    )

    # Train the baseline model
    torch.manual_seed(config.random_seed)
    ae = AutoEncoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
    )
    logging.info("Training the Base AutoEncoder...")
    ae_train_losses, ae_eval_losses = train(config, ae, train_loader, test_loader)

    # Train the tied weights model
    torch.manual_seed(config.random_seed)
    tied_ae = TiedAutoEncoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        hidden_dims=config.hidden_dims,
    )
    logging.info("Training the Tied AutoEncoder...")
    tied_ae_train_losses, tied_ae_eval_losses = train(
        config, tied_ae, train_loader, test_loader
    )

    # Plot loss curves for both models
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_loss_curves(ae_train_losses, ae_eval_losses, axes[0], "Base AutoEncoder Loss")
    plot_loss_curves(
        tied_ae_train_losses, tied_ae_eval_losses, axes[1], "Tied AutoEncoder Loss"
    )
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
