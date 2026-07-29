import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_swiss_roll(X: np.ndarray, t: np.ndarray, ax: Axes3D) -> None:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    sc = ax.scatter(x, y, z, c=t, cmap="viridis")
    ax.get_figure().colorbar(sc, shrink=0.6, pad=0.15, label="t")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
