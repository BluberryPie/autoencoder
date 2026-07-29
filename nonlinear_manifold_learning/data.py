import numpy as np
from numpy.random import Generator


def generate_swiss_roll(
    n_sample: int, noise: float, rng: Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = rng.uniform(low=np.pi * 1.5, high=np.pi * 4.5, size=n_sample)
    y = rng.uniform(low=0, high=10, size=n_sample)

    x = t * np.cos(t)
    z = t * np.sin(t)

    X = np.vstack(tup=[x, y, z]).T
    X += rng.standard_normal(size=X.shape) * noise

    return X, t, y
