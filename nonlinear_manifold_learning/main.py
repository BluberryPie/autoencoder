import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import Config
from data import generate_swiss_roll
from model import AutoEncoder
from visualize import plot_swiss_roll, plot_latent_space


def standardize(X: np.ndarray):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std
    return X, X_mean, X_std


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    config = Config()
    rng = np.random.default_rng(config.random_seed)
    torch.manual_seed(config.random_seed)
    X_raw, t, y = generate_swiss_roll(
        n_sample=config.num_training_samples, noise=config.noise_scale, rng=rng
    )

    X, X_mean, X_std = standardize(X_raw)
    X = torch.from_numpy(X).float()

    model = AutoEncoder(
        input_dim=X.shape[1], latent_dim=2, hidden_dims=config.hidden_dims
    )
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(config.num_train_iterations):
        X_reconstructed = model(X)
        loss = loss_fn(X, X_reconstructed)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % (config.num_train_iterations // 10) == 0:
            logging.info(
                f"[Iteraion {i + 1:>5}/{config.num_train_iterations}] -> Loss: {loss.item():.4f}"
            )

    with torch.no_grad():
        latents = model.encode(X).numpy()
        X_reconstructed = model(X).numpy()
        X_reconstructed = (X_reconstructed * X_std) + X_mean

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2)
    plot_swiss_roll(X_raw, t, ax1, label="t")
    plot_latent_space(latents, t, ax2, label="t")
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4)
    plot_swiss_roll(X_raw, y, ax3, label="y")
    plot_latent_space(latents, y, ax4, label="y")
    fig.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw={"projection": "3d"})
    plot_swiss_roll(X_raw, t, axes[0], label="t")
    plot_swiss_roll(X_reconstructed, t, axes[1], label="t")
    axes[0].set_title("Original")
    axes[1].set_title("Reconstructed")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
