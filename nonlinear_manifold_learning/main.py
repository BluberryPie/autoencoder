import matplotlib.pyplot as plt
import numpy as np

from data import generate_swiss_roll
from visualize import plot_swiss_roll


def main():
    rng = np.random.default_rng(seed=42)
    X, t, y = generate_swiss_roll(n_sample=1000, noise=0.1, rng=rng)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plot_swiss_roll(X, t, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
