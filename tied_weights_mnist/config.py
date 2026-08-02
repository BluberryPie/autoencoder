from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Config:
    random_seed: int = 42
    data_dir: str = "data"
    batch_size: int = 32
    input_dim: int = 28 * 28
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    num_epochs: int = 3
    learning_rate: float = 1e-3
