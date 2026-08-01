import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

mnist_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))]
)


def load_mnist(batch_size: int, data_dir: str) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=mnist_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=mnist_transform,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_dataloader, test_dataloader
