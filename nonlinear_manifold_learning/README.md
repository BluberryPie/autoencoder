# Nonlinear Manifold Learning with an Autoencoder

A small study project exploring how a vanilla (fully-connected) autoencoder learns to represent
a nonlinear 2D manifold(the classic Swiss roll), embedded in 3D space.

## Goal

Generate synthetic 3D data that actually lives on a 2D surface, train an autoencoder with a
2D bottleneck to compress and reconstruct it, and qualitatively inspect:

- Whether the learned 2D latent space recovers the manifold's true underlying structure.
- How that compares to the ground-truth generating coordinates.
- Whether reconstruction quality alone tells us anything about the quality of the learned
  representation.

## The data: Swiss roll

The Swiss roll has exactly two true degrees of freedom, `(t, y)`:

- `t`: position along the spiral, sampled from `Uniform(1.5π, 4.5π)`.
- `y`: position across the width of the sheet, sampled independently from `Uniform(0, 10)`.

These are embedded into 3D via an Archimedean spiral (radius grows linearly with angle):

```
x = t * cos(t)
z = t * sin(t)
y = y
```

Optional Gaussian noise is added to `(x, y, z)` after embedding. The `1.5π` starting offset
avoids overly cramped coils near the origin, and the `3π` span (1.5 full turns) is what makes
different coils of the roll pass close to each other in 3D — the reason this dataset defeats
linear methods like PCA: **points close in Euclidean 3D space can be far apart along the true
manifold**, so a purely linear projection cannot "unroll" it correctly.

## Model

A standard fully-connected autoencoder:

- Encoder: `3 → hidden_dims → 2` (2D bottleneck)
- Decoder: `2 → reversed(hidden_dims) → 3` (symmetric)
- Nonlinear activation between layers (no activation on the final output of either half) —
  without this, a stack of `Linear` layers collapses to a single linear map, making the model
  mathematically equivalent to PCA regardless of depth.

Inputs are standardized (zero mean, unit variance, per feature) before training and the
reconstruction is un-standardized back to the original scale for visualization.

Trained with full-batch gradient descent (Adam, MSE reconstruction loss) — the dataset is small
enough that mini-batching isn't necessary.

## Key concept: autoencoders are not identifiable

A plain reconstruction loss only requires that the latent code be *invertible* back to the
input — nothing forces it to match any particular "true" parametrization. Any smooth,
invertible reparametrization of the latent space (rotation, rescaling, warping) reconstructs
equally well. So the realistic goal isn't "recover `(t, y)` exactly," but rather: **does the
learned 2D space vary smoothly and consistently with the true manifold coordinates**, even if
reshaped into a different (but still valid) coordinate system?

## Results

### Original manifold vs. learned latent space

<img width="1087" height="386" alt="Image" src="https://github.com/user-attachments/assets/db378b56-06a5-4583-a998-84e44fbf8f13" />

The learned 2D latent space did **not** unroll into a flat rectangle (which is what a method
explicitly designed to preserve manifold/geodesic distances, like Isomap, would aim for).
Instead it formed a smooth, ring-like ("C" shaped) embedding — but critically, color (`t`)
varies smoothly and monotonically around it, meaning points close in the true parameter
stayed close in the learned code. This is a valid, if unexpected, unrolling.

### Coloring by the second factor, `y`

<img width="1089" height="362" alt="Image" src="https://github.com/user-attachments/assets/a33bc869-e9e0-46bc-ba2d-5d05c8fd42d5" />

Coloring the same latent space by `y` instead of `t` revealed that the network had effectively
rediscovered **polar coordinates**: `t` (the spiral position) maps to *angle* around the ring,
while `y` (the width) maps to *radius*. The two true independent generative factors were
cleanly separated into two orthogonal directions in latent space — just not the Cartesian
`(x, y)` layout one might naively expect.

### Reconstruction quality

<img width="1138" height="472" alt="Image" src="https://github.com/user-attachments/assets/c182846c-6a70-49b3-a64c-06da7da061b6" />

The reconstructed points are visually near-identical to the originals. This is expected rather
than surprising: reconstruction is evaluated on the same points the model trained on (no
held-out test set), and with enough capacity and training iterations on a small dataset, near-
perfect training reconstruction is close to guaranteed by the loss function itself. **It does
not, by itself, indicate that the latent space is well-structured** — that's exactly why the
qualitative latent-space plots above are the more informative artifacts here, not the
reconstruction error.

## Practical observations from tuning

- Smoother activations (e.g. `Tanh`) tend to produce visibly smoother-looking latent
  embeddings than `ReLU`, since `ReLU` networks are piecewise-linear (faceted) rather than
  everywhere-differentiable.
- Smaller hidden layers reduce the model's freedom to fold/distort the latent space arbitrarily which oftentimes produced better results(qualitatively).
- Different random seeds converged to meaningfully different latent
  layouts(no single "correct" answer is being targeted by the loss).

## Project structure

- `data.py` — Swiss roll data generation.
- `model.py` — configurable symmetric MLP autoencoder.
- `visualize.py` — 3D manifold and 2D latent-space plotting utilities.
- `config.py` — experiment hyperparameters.
- `main.py` — ties it all together: generate data, train, visualize.

## Running it

```bash
uv run python main.py
```
