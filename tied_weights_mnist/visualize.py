import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from torch.utils.data import Dataset, DataLoader

from model import AutoEncoder, TiedAutoEncoder


def plot_loss_curves(
    train_loss: list[float], eval_loss: list[float], ax: Axes, title: str
):
    ax.plot(train_loss, color="tab:blue")
    ax.plot(eval_loss, color="tab:orange")
    ax.set_title(title)


@torch.no_grad()
def plot_reconstructions(
    loader: DataLoader,
    ae: AutoEncoder,
    tied_ae: TiedAutoEncoder,
    num_samples_per_class: int = 10,
):
    dataset: Dataset = loader.dataset
    sample_indices: list[int] = []
    for class_idx in range(0, 10):
        sample_indices += (
            (dataset.targets == class_idx)
            .nonzero()[:num_samples_per_class]
            .flatten()
            .tolist()
        )
    X = torch.stack([loader.dataset[idx][0] for idx in sample_indices])
    # Switch the models to evaluation mode
    ae.eval()
    tied_ae.eval()

    # Generate reconstructions for both models
    preds_ae, preds_tied_ae = ae(X), tied_ae(X)

    # Reshape tensors for plotting
    X = X.reshape(10, num_samples_per_class, -1)
    preds_ae = preds_ae.reshape(10, num_samples_per_class, -1)
    preds_tied_ae = preds_tied_ae.reshape(10, num_samples_per_class, -1)

    # Plot
    fig = plt.figure(figsize=(14, 9))
    subfigs = fig.subfigures(nrows=5, ncols=2).flat
    for class_idx, subfig in enumerate(subfigs):
        subfig.suptitle(f"Class: {class_idx}")
        axes = subfig.subplots(nrows=3, ncols=num_samples_per_class)
        sources = [
            ("Original", X[class_idx]),
            ("AE", preds_ae[class_idx]),
            ("Tied AE", preds_tied_ae[class_idx]),
        ]
        for row, (model, images) in enumerate(sources):
            for col in range(num_samples_per_class):
                image = images[col].reshape(28, 28)
                axis = axes[row][col]
                axis.imshow(image, cmap="gray", vmin=0, vmax=1)
                if (col == 0) and (class_idx % 2 == 0):
                    axis.set_ylabel(model)
                axis.set_xticks([])
                axis.set_yticks([])

    plt.show()

    # Switch the models back to training mode
    ae.train()
    tied_ae.train()
