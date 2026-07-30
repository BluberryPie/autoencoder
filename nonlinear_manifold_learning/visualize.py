import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D


def plot_swiss_roll(X: np.ndarray, t: np.ndarray, ax: Axes3D, label: str) -> None:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    sc = ax.scatter(x, y, z, c=t, cmap="viridis")
    ax.get_figure().colorbar(sc, shrink=0.6, pad=0.15, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_latent_space(Z: np.ndarray, t: np.ndarray, ax: Axes, label: str) -> None:
    z1, z2 = Z[:, 0], Z[:, 1]
    sc = ax.scatter(z1, z2, c=t, cmap="viridis")
    ax.get_figure().colorbar(sc, shrink=0.6, pad=0.15, label=label)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_aspect("equal")
